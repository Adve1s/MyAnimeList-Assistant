## **MyAnimeList Assistant**

*A Python-based application that fetches and analyzes anime data using the MyAnimeList API.*

## **Features**  

✅ Fetches anime data from MyAnimeList API 📡  
✅ Shows **seasonal anime** and **top-rated anime** 🔺  
✅ Generates **personalized recommendations** based on MAL user data 🎯  
✅ Caches API responses for faster performance ⚡  
✅ Uses **multithreading** to ensure a smooth experience and prevent UI freezing 💡  
✅ Basic GUI using Tkinter 🎨  

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
   - If you don’t have an API key, [register on MyAnimeList](https://myanimelist.net/apiconfig)  

## **Usage**  

Run the program with:  

```sh
python main.py
```  

### **Using the Test Account (Optional)**  

By default, the program **pre-fills a public test account** to make testing easier.  
If you have your own MyAnimeList account, simply **delete the pre-filled username** and enter yours instead.  

### **Functions**  

- **Global Functions** *(No username required)*  
  - View **Top Anime of All Time**  
  - View **This Season’s Anime**  

- **Personalized Functions** *(Requires a MAL username)*  
  - Fetch your **anime list**  
  - Generate **personalized recommendations**  
  - Refresh data & clear cache  

## **Cache Files**  

The program stores API responses in these files:  

- `anime_list_cache.json` – Cached anime lists  
- `anime_details_cache.json` – Cached anime details  

*These files are included to speed up future queries.*  

## **Notes**  

⚠ **API key security:** Do **NOT** share your `.env` file publicly!  

---  

This README ensures your project is **clear, professional, and user-friendly**. 🚀  
