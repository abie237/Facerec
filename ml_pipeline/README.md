# ML Pipeline

All face detection, encoding, and classifier training/evaluation code lives
in this folder. **Run the scripts in this exact order** — each step depends
on the output of the previous one.

## Run order

```bash
# 1. Detect + encode every face in data/dataset/<student_name>/*.jpg
#    -> writes data/encodings/encodings.pkl
#    -> registers each student in database/attendance.db
python ml_pipeline/encode_faces.py

# 2. Load raw encodings, do a stratified train/test split,
#    train SVM + KNN, evaluate, export trained models
#    -> writes ml_pipeline/models/*.pkl
python ml_pipeline/train_classifier.py

# 3. Open the notebook for the full visual report
#    (confusion matrices, comparison charts, per-student metrics)
jupyter notebook ml_pipeline/evaluate_models.ipynb
```

Run all three from the **project root** (not from inside `ml_pipeline/`),
so the relative paths resolve correctly.

## Files

| File | Purpose |
|---|---|
| `encode_faces.py` | OpenCV reads each image; `face_recognition` detects the face and computes a 128-d embedding. Per-student embeddings are averaged and stored in the `students` table; raw per-image embeddings are saved separately for training. |
| `train_classifier.py` | Stratified train/test split on the raw embeddings, trains an SVM and a KNN classifier on the same split, evaluates both (accuracy/precision/recall/F1 + confusion matrix), exports both models. |
| `evaluate_models.ipynb` | Loads the exported models/metrics and produces the visual report: dataset composition chart, SVM vs KNN comparison chart, confusion matrix plots, full classification reports, and a written limitations discussion — this is what you screenshot for the thesis Results chapter. |
| `models/` | Exported `.pkl` artifacts: `svm_model.pkl`, `knn_model.pkl`, `label_encoder.pkl`, `evaluation_results.pkl`. The Streamlit app (`app/recognizer.py`) loads these directly — **retraining is only needed when students are added/removed**, not on every attendance session. |

## Known limitation (document in thesis)

Both `encode_faces.py` and `train_classifier.py` will run correctly with as
few as 2 images per student, but will print explicit warnings if any
student has fewer than 3 images, since the held-out test set for that
student will then contain only a single image. The thesis's own
recommended minimum is 10 images per student; the real deployment dataset
(~3 students, 2-3 images each) falls below this. The pipeline is
methodologically sound regardless — this is a **data volume** limitation,
not a code or method limitation, and should be framed that way in the
Discussion/Limitations chapter.
