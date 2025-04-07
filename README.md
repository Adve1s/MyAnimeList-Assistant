## **MyAnimeList Assistant**

_A Python-based application that fetches, analyzes, and recommends anime using the MyAnimeList API and machine learning._

## **Features**

✅ Fetches anime data from MyAnimeList API 📡  
✅ Shows **seasonal anime** and **top-rated anime** 🔺  
✅ Generates **personalized recommendations** using two methods:

- Based on MAL user ratings & community recommendations 🎯
- Using **machine learning** for advanced personalized predictions 🧠
  ✅ Caches API responses for faster performance ⚡  
  ✅ Uses **multithreading** to ensure a smooth experience and prevent UI freezing 💡  
  ✅ Basic GUI using Tkinter 🎨

## **Machine Learning Features**

The application now uses a RandomForest model to provide intelligent anime recommendations:

🔍 **Feature-based Analysis**: Analyzes anime characteristics including genres, studios, seasons, and more  
🎭 **Genre Preferences**: Learns your genre preferences from your ratings  
📊 **Statistical Learning**: Uses community scores, popularity metrics, and episode counts as features  
💯 **Personalized Scores**: Predicts your likely rating for anime you haven't watched yet  
📈 **Tuned Performance**: Automatically tunes hyperparameters for optimal recommendations

## **Installation**

1. **Clone this repository**:
   ```sh
   git clone https://github.com/Adve1s/MyAnimeList-Assistant.git
   cd MyAnimeList-Assistant
   ```
2. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```
3. **Set up API credentials**:
   - Rename `.env.example` to `.env`
   - Open `.env` and add your MyAnimeList API key:
     ```ini
     MAL_CLIENT_ID=your_client_id_here
     ```
   - If you don't have an API key, [register on MyAnimeList](https://myanimelist.net/apiconfig)

## **ML Dependencies**

For machine learning features, you'll need these additional libraries:

- pandas
- scikit-learn

Install them using:

```sh
pip install pandas scikit-learn
```

## **Usage**

Run the program with:

```sh
python main.py
```

### **Using the Test Account (Optional)**

By default, the program **pre-fills a public test account** to make testing easier.  
If you have your own MyAnimeList account, simply **delete the pre-filled username** and enter yours instead.

### **Functions**

- **Global Functions** _(No username required)_

  - View **Top Anime of All Time**
  - View **This Season's Anime**

- **Personalized Functions** _(Requires a MAL username)_
  - Fetch your **anime list**
  - Generate **traditional recommendations** based on your ratings
  - Generate **ML-powered recommendations** for personalized predictions
  - Update global ranking list data
  - Refresh data & clear cache

## **Cache Files**

The program stores API responses in these files:

- `anime_list_cache.json` – Cached anime lists
- `anime_details_cache.json` – Cached anime details
- `global_rankings_cache.json` – Cached global rankings data

_These files are included to speed up future queries._

## **How the ML Recommender Works**

The machine learning recommender:

1. Extracts features from anime you've completed and rated
2. Trains a RandomForest regression model on your rating patterns
3. Optimizes model parameters automatically
4. Predicts scores for anime you haven't watched
5. Presents recommendations sorted by predicted score

The model considers factors like:

- Anime genres (action, romance, comedy, etc.)
- Studios that produced the anime
- Release season and year
- Episode count
- Community ratings and popularity

## **Notes**

⚠ **API key security:** Do **NOT** share your `.env` file publicly!  
⚠ **First run:** The first run may take longer as it builds necessary caches.
⚠ **ML Requirements:** Ensure you have pandas and scikit-learn installed for ML features.
