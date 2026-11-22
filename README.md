# CS471 Final Project: HMM Location Predictor

Predicts where a moving object will be, using a Hidden Markov Model and the forward algorithm on noisy 3D location reports. Plain Python, no libraries.

## 📘 Start here: [Tutorial (PDF)](checkpoint/HMM_Predictor_Tutorial.pdf)

## Documentation

- **[HMM Predictor Tutorial](checkpoint/HMM_Predictor_Tutorial.pdf)**: walkthrough of how to use the predictor
- **[HMM Predictor Documentation](checkpoint/HMM_Predictor_Documentation.pdf)**: full write-up of the model and implementation
- **[Final Project Guide](471_Final_Project_Guide.pdf)**: course assignment guide

## Quick start

```
cd checkpoint
python generate_tracks.py           # makes test_track.csv plus 50 test tracks and answer keys
python hmm_predict.py test_track.csv
```

The input CSV needs a header row `t,x,y,z` (t in seconds). The program prints a current position estimate for each report, then predicts the position 2, 5 and 10 seconds after the last report, with uncertainty.

## Files

| Path | What it is |
|---|---|
| [checkpoint/hmm_predict.py](checkpoint/hmm_predict.py) | The HMM predictor |
| [checkpoint/generate_tracks.py](checkpoint/generate_tracks.py) | Generates synthetic test tracks and answer keys |
| [checkpoint/predictions/](checkpoint/predictions/) | Predictor output for the 50 test tracks |
| [checkpoint/AI_Test.zip](checkpoint/AI_Test.zip) | Archive of the test predictions |
