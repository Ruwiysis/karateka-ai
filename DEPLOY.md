# 🥋 Karateka — Deploy to Railway
## VS Code → GitHub → Railway → Live Global URL (~5 min, FREE)

---

## WHY MEDIAPIPE IS PINNED TO 0.10.14
mediapipe versions above 0.10.14 removed `mp.solutions` API.
The server uses `mp.solutions.pose` — **do NOT upgrade mediapipe**.
The requirements.txt already pins it correctly.

---

## STEP 1 — Test locally

```bash
pip install -r requirements.txt
python server.py
```
Open → http://localhost:5000
Allow camera → confirm it works ✅

---

## STEP 2 — Create GitHub repo

1. github.com → **New repository**
2. Name: `karateka` — Public — no README
3. Click **Create repository**

---

## STEP 3 — Push from VS Code Terminal

```bash
git init
git add .
git commit -m "Karateka initial deploy"
git remote add origin https://github.com/YOURNAME/karateka.git
git branch -M main
git push -u origin main
```

---

## STEP 4 — Deploy on Railway

1. Go to **railway.app** → Login with GitHub
2. Click **New Project** → **Deploy from GitHub repo**
3. Select your `karateka` repo
4. Railway reads `railway.json` automatically — no settings needed
5. Click **Deploy** → wait ~3 minutes for build

**Your live URL:** `https://karateka-production.up.railway.app`

---

## STEP 5 — Done!

Send the URL to anyone.
✅ Works on desktop & mobile
✅ Camera works for every user on their own device
✅ HTTPS automatic (required for camera permissions)
✅ No sleep (unlike Render free tier)

---

## Update later

```bash
git add .
git commit -m "update"
git push
```
Railway auto-redeploys on every push (~2 min).

---

## Verify it's working

Visit: `https://your-url.railway.app/health`
Should return:
```json
{"model": true, "moves": ["Mae_Geri", ...], "status": "ok"}
```
