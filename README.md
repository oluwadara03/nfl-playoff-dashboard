GitHub: https://github.com/oluwadara03/nfl-playoff-dashboard


Run with Docker:
docker build -t nfl-playoff-predictor .  

docker run --rm -p 3000:3000 nfl-playoff-predictor  

Then open localhost link: http://localhost:3000  



Run locally:
pip install -r requirements.txt  

streamlit run app/dashboard.py  



How it works:
* One row per team per season (features: win %, points for/against per game, point differential, made playoffs). 
Made playoffs means a team appeared in any postseason game.
* Training pairs: season N stats -> did the team make the playoffs in season N+1. 2023->2024 and 2024->2025 give 64 labelled rows. The 2026 prediction uses 2025 stats.
* Chronological split: train 2023->2024, test 2024->2025 (was only used once); the final model is refit on both pairs.
* Models: Logistic Regression (point differential only), Random Forest with monotonic constraints, and a naive repeat-last-year baseline. Projected wins: linear regression.



Main folder structure:

<img width="300" height="458" alt="image" src="https://github.com/user-attachments/assets/b9bbe315-a3ff-408c-b1f0-b13fa3eddb97" />

