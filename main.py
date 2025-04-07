import requests
import json
import os
import io
import time
import threading
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from dotenv import load_dotenv
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageTk

class AnimeMLRecommender:
    """
    Machine learning recommender for anime based on user ratings.
    Provides personalized predictions using a RandomForest model.
    """
    
    def __init__(self, parent_app):
        """
        Initialize the recommender with a reference to the parent app.
        
        Args:
            parent_app: Reference to the parent MyAnimeListApp instance
        """
        self.parent = parent_app  # Store reference to parent application
        self.model = None         # Will hold the trained ML model
        self.trained = False      # Flag to track if model has been trained
        
        # Import required ML libraries with error handling
        try:
            import pandas as pd
            from sklearn.ensemble import RandomForestRegressor
            self.pd = pd
            self.RandomForestRegressor = RandomForestRegressor
            self.ml_available = True  # Flag indicating ML libraries are available
            print("ML libraries successfully imported.")
        except ImportError:
            print("ML libraries not available. Please install pandas and scikit-learn.")
            self.ml_available = False
    
    def prepare_data(self):
        """
        Process anime data and prepare it for training by extracting features.
        Converts user's anime list and details into a format suitable for ML.
        
        Returns:
            tuple: (X, y) feature matrix and target vector, or (None, None) if error
        """
        if not self.ml_available:
            print("ML libraries not available.")
            return None, None
            
        try:
            # Create lists to store data for DataFrame construction
            data = []
            
            # Loop through completed anime with scores to create training data
            for anime in self.parent.anime_list:
                # Skip if not completed or no score (can't learn from unrated anime)
                if anime["list_status"]["status"] != "completed" or anime["list_status"]["score"] == 0:
                    continue
                
                anime_id = str(anime["node"]["id"])
                
                # Skip if no details available in cache
                if anime_id not in self.parent.anime_details:
                    continue
                
                details = self.parent.anime_details[anime_id]["details"]
                
                # Create a row with basic information about this anime
                row = {
                    "id": anime_id,
                    "title": anime["node"]["title"],
                    "user_score": anime["list_status"]["score"],  # This will be our target variable
                    "mean_score": details.get("mean", 0),         # Average community rating
                    "popularity": details.get("popularity", 0),   # Popularity metric
                    "num_episodes": details.get("num_episodes", 0)  # Episode count
                }
                
                # Add genre information using one-hot encoding
                # Each genre becomes a binary feature (1=has genre, 0=doesn't have genre)
                genres = details.get("genres", [])
                for genre in genres:
                    genre_name = f"genre_{genre['name'].replace(' ', '_')}"
                    row[genre_name] = 1
                
                # Add studio information if available
                studios = details.get("studios", [])
                for studio in studios:
                    studio_name = f"studio_{studio['name'].replace(' ', '_')}"
                    row[studio_name] = 1
                
                # Add start season information if available
                # These can be strong indicators of anime style and quality
                start_season = details.get("start_season", {})
                if start_season:
                    row["year"] = start_season.get("year", 0)
                    season = start_season.get("season", "")
                    if season:
                        row[f"season_{season}"] = 1
                
                data.append(row)
            
            # Convert list of dictionaries to pandas DataFrame
            df = self.pd.DataFrame(data)
            
            # Fill missing values for categorical features with zeros
            # This ensures one-hot encoded features are properly represented
            for col in df.columns:
                if col.startswith("genre_") or col.startswith("studio_") or col.startswith("season_"):
                    df[col] = df[col].fillna(0)
            
            # Ensure numeric values are appropriate by filling NaN with median values
            numeric_cols = ["mean_score", "rank", "popularity", "num_episodes", "year"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].median())
            
            # Define features (X) and target (y) for machine learning
            y = df["user_score"]  # What we're trying to predict
            X = df.drop(["id", "title", "user_score"], axis=1)  # Features used for prediction
            
            # Store feature names for later use in predictions and importance analysis
            self.feature_names = X.columns.tolist()
            
            print(f"Prepared data with {len(X)} anime and {len(X.columns)} features.")
            return X, y
            
        except Exception as e:
            print(f"Error preparing data: {e}")
            return None, None
    
    def tune_and_train(self):
            """
            Tune hyperparameters and train the model in one step.
            
            Returns:
                bool: Success status
            """
            if not self.ml_available:
                print("ML libraries not available.")
                return False
            
            try:
                from sklearn.model_selection import GridSearchCV
                
                # Prepare data
                X, y = self.prepare_data()
                if X is None or len(X) < 4:
                    print("Not enough rated anime to train model (need at least 4).")
                    return False
                
                print(f"Training model with {len(X)} anime...")
                
                # Define parameter grid
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                
                # Create base model
                base_model = self.RandomForestRegressor(random_state=42)
                
                # Setup grid search with cross-validation
                print("Tuning hyperparameters (this may take some time)...")
                grid_search = GridSearchCV(
                    estimator=base_model,
                    param_grid=param_grid,
                    cv=3,  # 3-fold cross-validation
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,  # Use all CPU cores
                    verbose=1
                )
                
                # Perform the search and training in one step
                grid_search.fit(X, y)
                
                # Get best model and parameters
                self.model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                best_score = -grid_search.best_score_
                
                # Print results
                print(f"Best parameters: {best_params}")
                print(f"Best MSE: {best_score:.4f}")
                print(f"Root MSE: {best_score ** 0.5:.4f}")
                
                # Set trained flag
                self.trained = True
                
                # Print feature importance
                self._print_feature_importance()
                
                return True
                
            except Exception as e:
                print(f"Error during tuning and training: {e}")
                return False

    def _print_feature_importance(self):
        """
        Print the most important features in the model.
        Helps understand which anime characteristics most influence the predictions.
        """
        if not self.trained or self.model is None:
            return
            
        # Get feature importance values from the trained model
        importance = self.model.feature_importances_
        
        # Create a list of (feature, importance) tuples
        feature_importance = list(zip(self.feature_names, importance))
        
        # Sort by importance (descending)
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Print top 10 features with readable names
        print("\nTop 10 Most Important Features:")
        for feature, score in feature_importance[:10]:
            # Make feature name more readable for display
            readable_name = feature
            if feature.startswith("genre_"):
                readable_name = feature.replace("genre_", "").replace("_", " ")
            elif feature.startswith("studio_"):
                readable_name = feature.replace("studio_", "").replace("_", " ")
            elif feature.startswith("season_"):
                readable_name = feature.replace("season_", "").capitalize()
                
            print(f"{readable_name}: {score:.4f}")
    
    def predict(self, anime_id):
        """
        Predict rating for a single anime.
        
        Args:
            anime_id (str): ID of the anime to predict
        
        Returns:
            float or None: Predicted rating (1-10 scale) or None if prediction fails
        """
        if not self.trained or self.model is None:
            print("Model not trained yet.")
            return None
            
        try:
            # Get anime details from cache
            if anime_id not in self.parent.anime_details:
                print(f"No details for anime ID {anime_id}")
                return None
                
            details = self.parent.anime_details[anime_id]["details"]
            
            # Create feature row matching the format used in training
            row = {
                "mean_score": details.get("mean", 0),
                "popularity": details.get("popularity", 0),
                "num_episodes": details.get("num_episodes", 0)
            }
            
            # Add genre information
            genres = details.get("genres", [])
            for genre in genres:
                genre_name = f"genre_{genre['name'].replace(' ', '_')}"
                row[genre_name] = 1
            
            # Add studio information
            studios = details.get("studios", [])
            for studio in studios:
                studio_name = f"studio_{studio['name'].replace(' ', '_')}"
                row[studio_name] = 1
            
            # Add start season information
            start_season = details.get("start_season", {})
            if start_season:
                row["year"] = start_season.get("year", 0)
                season = start_season.get("season", "")
                if season:
                    row[f"season_{season}"] = 1
            
            # Convert to DataFrame for prediction
            df = self.pd.DataFrame([row])
            
            # Ensure all training features exist (even if they're zero)
            # Missing features would cause prediction errors
            for feature in self.feature_names:
                if feature not in df.columns:
                    df[feature] = 0
            
            # Select only features used in training to maintain consistency
            X = df[self.feature_names]
            
            # Make prediction using the trained model
            prediction = self.model.predict(X)[0]
            
            # Round to one decimal place for readability
            return round(prediction, 1)
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def predict_multiple(self, anime_ids):
        """
        Predict ratings for multiple anime efficiently.
        
        Args:
            anime_ids (list): List of anime IDs to predict ratings for
        
        Returns:
            dict: Dictionary mapping anime IDs to predicted ratings
        """
        predictions = {}
        for anime_id in anime_ids:
            pred = self.predict(anime_id)
            if pred is not None:
                predictions[anime_id] = pred
        
        return predictions

    def get_ml_recommendations(self, top_n=20, anime_list=None):
        """
        Generate ML-based recommendations for anime.
        Uses model to predict scores for unwatched anime or a specific list.
        
        Args:
            top_n (int): Number of recommendations to return
            anime_list (list, optional): Specific list of anime IDs to predict from.
                                        If None, predicts from all unwatched anime.
            
        Returns:
            list: Recommendations as (anime_id, score, title) tuples, sorted by predicted score
        """
        if not self.trained or self.model is None:
            print("Model not trained yet.")
            return []
        
        try:
            # Determine which anime IDs to predict
            if anime_list is not None:
                # Use the provided custom list
                candidate_ids = [str(anime_id) for anime_id in anime_list]
                print(f"Generating recommendations from {len(candidate_ids)} specified anime.")
            else:
                # Get all anime IDs the user has already seen (any status)
                seen_anime_ids = {
                    int(anime["node"]["id"]) for anime in self.parent.anime_list if anime["list_status"]["status"] == "completed"
                }
                # Get all anime IDs from details cache that user hasn't seen
                candidate_ids = [
                    anime_id for anime_id in self.parent.anime_details
                    if int(anime_id) not in seen_anime_ids
                ]
            
            # Filter to ensure we only use IDs with available details
            valid_candidate_ids = [
                anime_id for anime_id in candidate_ids
                if anime_id in self.parent.anime_details
            ]
            
            if len(valid_candidate_ids) == 0:
                print("No valid anime found for recommendations.")
                return []
            
            # Process all candidate anime in batches to avoid DataFrame fragmentation
            all_rows = []
            all_ids = []
            all_titles = []
            processed_anime_ids = set()

            for anime_id in valid_candidate_ids:
                # Get anime details
                details = self.parent.anime_details[anime_id]["details"]
                title = details.get("title", "Unknown")
                if int(anime_id) in processed_anime_ids or int(anime_id) in seen_anime_ids:
                    continue
                
                # Create a feature row matching the format used in training
                row = {}
                
                # Create a feature row matching the format used in training
                row = self._extract_features_from_details(details)
                
                all_rows.append(row)
                all_ids.append(anime_id)
                all_titles.append(title)
                processed_anime_ids.add(int(anime_id))

            if anime_list is None:
                for batch in self.parent.ranking_list:
                    for anime in batch:
                        # Get anime details
                        details = anime["node"]
                        title = anime["node"]["title"]
                        anime_id = anime["node"]["id"]
                        if int(anime_id) in seen_anime_ids or int(anime_id) in processed_anime_ids:
                            continue
                        # Create a feature row matching the format used in training
                        row = {}
                        
                        # Create a feature row matching the format used in training
                        row = self._extract_features_from_details(details)
                        
                        all_rows.append(row)
                        all_ids.append(str(anime_id))
                        all_titles.append(title)
                        processed_anime_ids.add(int(anime_id))

            # Create DataFrame 
            df = self.pd.DataFrame(all_rows)

            # Ensure all expected features exist with correct order
            # This is the critical part that ensures feature name match
            for feature in self.feature_names:
                if feature not in df.columns:
                    df[feature] = 0
            
            # Reorder columns to match exactly the order used during training
            df = df[self.feature_names]
            
            # Make predictions
            if len(df) > 0:
                # Make predictions on the entire batch
                predictions = self.model.predict(df)
                # Create list of (anime_id, score, title) tuples
                results = []
                for i, pred in enumerate(predictions):
                    results.append((all_ids[i], round(pred, 1), all_titles[i]))
                # Sort by predicted score (descending)
                results.sort(key=lambda x: x[1], reverse=True)
                print(f"Generating recommendations from {len(results)} anime.")
                # Return top N recommendations
                return results[:top_n]
            else:
                print("No valid anime data found for predictions.")
                return []
                
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return []
        
    def _extract_features_from_details(self, details):
        """
        Extract ML features from anime details.
        Helper method to reduce code duplication.
        
        Args:
            details (dict): Anime details dictionary
            
        Returns:
            dict: Feature dictionary for the anime
        """
        row = {}
        
        # Add basic numeric features
        row["mean_score"] = details.get("mean", 0)
        row["rank"] = details.get("rank", 0)
        row["popularity"] = details.get("popularity", 0)
        row["num_episodes"] = details.get("num_episodes", 0)
        
        # Add genre information
        genres = details.get("genres", [])
        for genre in genres:
            genre_name = f"genre_{genre['name'].replace(' ', '_')}"
            row[genre_name] = 1
        
        # Add studio information
        studios = details.get("studios", [])
        for studio in studios:
            studio_name = f"studio_{studio['name'].replace(' ', '_')}"
            row[studio_name] = 1
        
        # Add start season information
        start_season = details.get("start_season", {})
        if start_season:
            row["year"] = start_season.get("year", 0)
            season = start_season.get("season", "")
            if season:
                season_name = f"season_{season}"
                row[season_name] = 1
                
        return row
        
# Documentation reference: https://myanimelist.net/apiconfig/references/api/v2

class MyAnimeListApp:
    # Constants for caching
    ANIME_LIST_CACHE = "anime_list_cache.json"  # File to cache anime list data
    ANIME_DETAILS_CACHE_FILE = "anime_details_cache.json"  # File to cache anime details
    ANIME_GLOBAL_RANKINGS_FILE = "global_rankings_cache.json" # File to cache global rankings
    CACHE_EXPIRY_HOURS = 720  # Cache expiry duration in hours

    def __init__(self, root):
        """Initialize the GUI application and load configuration."""
        self.root = root
        self.root.title("MyAnimeList GUI")
        self.root.geometry("400x300")
        # Load environment variables from .env file
        load_dotenv()
        self.CLIENT_ID = os.getenv("MAL_CLIENT_ID")  # Client ID for API 
        self.CLIENT_SECRET = os.getenv("MAL_CLIENT_SECRET")  # Client secret for API (not currently used) 

        # Check if required environment variables are loaded
        if not self.CLIENT_ID:
            raise ValueError("CLIENT_ID not found. Please add it to your .env file.")
        
        # Initialize application data structures
        self.anime_list = []  # Store fetched anime list
        self.anime_details = {}  # Cache for anime details
        self.recommendation_list = {}  # Store recommendations
        self.recommendation_list_train = {}  # Store training recommendations for ML
        self.sorted_recommendations = []  # List for sorted recommendations
        self.ranking_list = []  #List for global rankings
        self.lock = threading.Lock()  # Thread safety lock

        self.ml_recommender = AnimeMLRecommender(self)

        # Create GUI components
        # Username input label
        self.username_label = tk.Label(root, text="Enter Username:")
        self.username_label.pack(pady=5)

        # Pre-fill username with a default value
        default_username = "test1234554"  # Example usernames: test1234554, AoIv315
        self.username_var = tk.StringVar(value=default_username)
        self.username_entry = tk.Entry(root, textvariable=self.username_var, width=30)
        self.username_entry.pack(pady=5)

        # Create buttons for user actions
        self.username_button = tk.Button(root, text="Functions with Username", command=self.functions_with_username)
        self.username_button.pack(pady=10)

        self.global_button = tk.Button(root, text="Global Functions", command=self.open_global_functions_window)
        self.global_button.pack(pady=10)

    def functions_with_username(self):
        """Execute username-specific functions after validating the input."""
        username = self.username_entry.get().strip()
        if username:
            self.username = username  # Save username for use across functions
            # Fetch the anime list using a threaded approach and handle success with a callback
            self.get_anime_list_threaded(self.on_get_anime_list_success)
        else:
            # Show a warning if the username field is empty
            messagebox.showwarning("Input Error", "Please enter a username.")

    def keep_on_top(self):
        """Ensure the second window stays on top temporarily."""
        self.second_window.attributes('-topmost', True)
        self.second_window.update_idletasks()
        self.second_window.attributes('-topmost', False)

    def on_get_anime_list_success(self):
        """Handle actions after successfully fetching the anime list."""
        # Create a new window for user-specific functions
        self.second_window = new_window = tk.Toplevel(self.root)
        new_window.title(f"Functions for {self.username}")
        self.second_window.geometry("400x300")

        # Display current username
        username_label = tk.Label(new_window, text=f"Current username: {self.username}")
        username_label.pack(pady=5)

        # Add buttons for various actions in the second window
        button1 = tk.Button(new_window, text="Generate Recommendations", command=self.create_recommendations_threaded)
        button1.pack(pady=5)

        button2 = tk.Button(new_window, text="Showcase Full List", command=self.showcase_list_threaded)
        button2.pack(pady=5)

        button3 = tk.Button(new_window, text="Clear All Cache", command=self.clear_all_cache)
        button3.pack(pady=5)

        # Add this button to your second window
        button4 = tk.Button(self.second_window, text="ML recommendations", command=self.create_ML_recommendations_threaded)
        button4.pack(pady=5)

        button5 = tk.Button(self.second_window, text="Update global ranking list", command=self.get_ranked_anime_api_threaded)
        button5.pack(pady=5)

    def open_global_functions_window(self):
        """
        Open a new window for global functions: Seasonal Anime and Rankings.
        """
        self.global_window = global_window = tk.Toplevel(self.root)
        global_window.title("Global Functions")
        global_window.geometry("400x300")

        # Button for Seasonal Anime
        seasonal_button = tk.Button(
            global_window,
            text="Seasonal Anime",
            command=self.show_this_seasons_anime_threaded
        )
        seasonal_button.pack(pady=20)

        # Button for Rankings
        rankings_button = tk.Button(
            global_window,
            text="Rankings",
            command=self.show_anime_rankings_threaded
        )
        rankings_button.pack(pady=20)

    def get_seasonal_anime(self, year=None, month=None, fields="", limit=50):
        """
        Fetch seasonal anime for the current year and season.

        Args:
            fields (str): Fields to include in the API response (e.g., "title,genres,mean").
            limit (int): Number of results to fetch (default is 100).

        Returns:
            list: A list of seasonal anime, or None if the request fails.
        """
        # Determine the current year and season
        from datetime import datetime

        current_date = datetime.now()
        if year == None:
            year = current_date.year
        if month == None:
            month = current_date.month

        # Determine season based on the current month
        if month in [1, 2, 3]:
            season = "winter"
        elif month in [4, 5, 6]:
            season = "spring"
        elif month in [7, 8, 9]:
            season = "summer"
        else:
            season = "fall"

        base_url = f"https://api.myanimelist.net/v2/anime/season/{year}/{season}"
        headers = {"X-MAL-CLIENT-ID": self.CLIENT_ID}
        params = {
            "limit": limit,
            "fields": fields
        }

        try:
            response = requests.get(base_url, headers=headers, params=params, timeout=10)
            response.raise_for_status()

            # Parse the JSON response
            if "application/json" in response.headers.get("Content-Type", ""):
                data = response.json()
                return data.get("data", [])  # Return the list of seasonal anime
            else:
                messagebox.showerror("Error", "Unexpected content type in API response.")
                return None
        except requests.exceptions.Timeout:
            messagebox.showerror("Error", "API request timed out. Try again later.")
            return None
        except requests.exceptions.RequestException as e:
            messagebox.showerror("Error", f"Request failed: {e}")
            return None
        except json.JSONDecodeError:
            messagebox.showerror("Error", "Failed to decode JSON response.")
            return None

    def show_this_seasons_anime(self):
        """
        Display the seasonal anime list in a new scrollable window with thumbnails and details.
        """
        fields = "title,main_picture,synopsis,genres,mean"  # Fields to fetch
        seasonal_anime = self.get_seasonal_anime(fields=fields)

        if not seasonal_anime:
            messagebox.showerror("Error", "Failed to fetch seasonal anime.")
            return

        # Create a scrollable window to display the seasonal anime
        showcase_window = tk.Toplevel(self.root)
        showcase_window.title("Seasonal Anime")
        showcase_window.geometry("800x600")
        canvas = tk.Canvas(showcase_window)
        scrollbar = tk.Scrollbar(showcase_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        # Update the canvas scrolling region dynamically
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bind mouse wheel scrolling to the canvas
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-1 * (e.delta // 120), "units"))

        # Store fetched images in a dictionary
        images = {}
        with ThreadPoolExecutor(max_workers=20) as executor:
            for anime, future in zip(seasonal_anime, executor.map(self.fetch_image, seasonal_anime)):
                images[anime['node']['id']] = future

        # Populate the scrollable frame with anime details
        for anime in seasonal_anime:
            frame = tk.Frame(scrollable_frame, pady=10, relief="groove", bd=2)
            frame.pack(fill="x", padx=10)

            # Display the thumbnail image at the top if available
            img = images.get(anime['node']['id'])
            if img:
                img_tk = ImageTk.PhotoImage(img)
                tk.Label(frame, image=img_tk).pack(side="top", pady=5)
                frame.image = img_tk  # Prevent image from being garbage collected

            # Display the anime's details
            details = (
                f"Title: {anime['node']['title']}\n"
                f"Score: {anime['node'].get('mean', 'N/A')}\n"
                f"Genres: {', '.join(g['name'] for g in anime['node'].get('genres', []))}\n\n"
                f"Synopsis: {anime['node'].get('synopsis', 'No synopsis available.')}"
            )
            tk.Label(
                frame, text=details, font=("Arial", 10), anchor="w", justify="left", wraplength=700
            ).pack(fill="x", pady=5)

            # Add a horizontal separator
            separator = ttk.Separator(scrollable_frame, orient="horizontal")
            separator.pack(fill="x", pady=10)

        # Add a button to close the window
        close_button = tk.Button(showcase_window, text="Close", command=showcase_window.destroy)
        close_button.pack(pady=10, side="bottom")

    def show_this_seasons_anime_threaded(self):
        """
        Start a thread to get seasonal anime and display them without blocking the GUI.
        """
        threading.Thread(target=self._show_this_seasons_anime_task).start()

    def _show_this_seasons_anime_task(self):
        """
        Task to generate recommendations and display the list in a thread-safe manner.
        """
        try:
            window = self.global_window  # Reference to the secondary window
            self.set_cursor_loading(window)  # Set cursor to loading
            with self.lock:
                self.show_this_seasons_anime()  # Get and display the list
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            self.reset_cursor(window)  # Reset cursor to normal

    def get_anime_rankings(self,ranking_type="all",limit=50, offset=0, fields="",):
        """
        Fetch anime rankings from the MyAnimeList API.

        Args:
            fields (str): Fields to include in the API response (e.g., "title,rank,mean").
            limit (int): Number of results to fetch (default is 50).
            ranking_type (str): Type of ranking to fetch (e.g., "all", "airing", "upcoming").

        Returns:
            list: A list of anime rankings, or None if the request fails.
        """
        base_url = "https://api.myanimelist.net/v2/anime/ranking"
        headers = {"X-MAL-CLIENT-ID": self.CLIENT_ID}
        params = {
            "ranking_type": ranking_type,
            "limit": limit,
            "fields": fields,
            "offset": offset,
            "nsfw": "true"
        }

        try:
            response = requests.get(base_url, headers=headers, params=params, timeout=10)
            response.raise_for_status()

            # Parse the JSON response
            if "application/json" in response.headers.get("Content-Type", ""):
                data = response.json()
                return data.get("data", [])  # Return the list of rankings
            else:
                messagebox.showerror("Error", "Unexpected content type in API response.")
                return None
        except requests.exceptions.Timeout:
            messagebox.showerror("Error", "API request timed out. Try again later.")
            return None
        except requests.exceptions.RequestException as e:
            messagebox.showerror("Error", f"Request failed: {e}")
            return None
        except json.JSONDecodeError:
            messagebox.showerror("Error", "Failed to decode JSON response.")
            return None

    def show_anime_rankings(self):
        """
        Display the anime rankings in a new scrollable window with thumbnails and details.
        """
        fields = "title,main_picture,synopsis,rank,mean,genres"  # Fields to fetch
        anime_rankings = self.get_anime_rankings(fields=fields)

        if not anime_rankings:
            messagebox.showerror("Error", "Failed to fetch anime rankings.")
            return

        # Create a scrollable window to display the anime rankings
        showcase_window = tk.Toplevel(self.root)
        showcase_window.title("Anime Rankings")
        showcase_window.geometry("800x600")
        canvas = tk.Canvas(showcase_window)
        scrollbar = tk.Scrollbar(showcase_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        # Update the canvas scrolling region dynamically
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bind mouse wheel scrolling to the canvas
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-1 * (e.delta // 120), "units"))

        # Store fetched images in a dictionary
        images = {}
        with ThreadPoolExecutor(max_workers=20) as executor:
            for anime, future in zip(anime_rankings, executor.map(self.fetch_image, anime_rankings)):
                images[anime['node']['id']] = future

        # Populate the scrollable frame with anime details
        for anime in anime_rankings:
            frame = tk.Frame(scrollable_frame, pady=10, relief="groove", bd=2)
            frame.pack(fill="x", padx=10)

            # Display the thumbnail image at the top if available
            img = images.get(anime['node']['id'])
            if img:
                img_tk = ImageTk.PhotoImage(img)
                tk.Label(frame, image=img_tk).pack(side="top", pady=5)
                frame.image = img_tk  # Prevent image from being garbage collected

            # Display the anime's details
            details = (
                f"Title: {anime['node']['title']}\n"
                f"Rank: {anime['node'].get('rank', 'N/A')} | Score: {anime['node'].get('mean', 'N/A')}\n"
                f"Genres: {', '.join(g['name'] for g in anime['node'].get('genres', []))}\n\n"
                f"Synopsis: {anime['node'].get('synopsis', 'No synopsis available.')}")
            tk.Label(
                frame, text=details, font=("Arial", 10), anchor="w", justify="left", wraplength=700
            ).pack(fill="x", pady=5)

            # Add a horizontal separator
            separator = ttk.Separator(scrollable_frame, orient="horizontal")
            separator.pack(fill="x", pady=10)

        # Add a button to close the window
        close_button = tk.Button(showcase_window, text="Close", command=showcase_window.destroy)
        close_button.pack(pady=10, side="bottom")

    def show_anime_rankings_threaded(self):
        """
        Start a thread to get seasonal anime and display them without blocking the GUI.
        """
        threading.Thread(target=self._show_anime_rankings_task).start()

    def _show_anime_rankings_task(self):
        """
        Task to generate recommendations and display the list in a thread-safe manner.
        """
        try:
            window = self.global_window  # Reference to the secondary window
            self.set_cursor_loading(window)  # Set cursor to loading
            with self.lock:
                self.show_anime_rankings()  # Get and display the list
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            self.reset_cursor(window)  # Reset cursor to normal

    def get_anime_list_threaded(self, callback=None):
        """Fetch the anime list in a separate thread and trigger a callback on success."""
        def task():
            success = self._get_anime_list_task()
            with self.lock:
                self.get_anime_details()
                self.create_recommendations()
                self.get_ranked_anime()
                self.reset_cursor()  # Reset cursor
            if success and callback:
                # Use root's after method to call the callback on the main thread
                self.root.after(0, callback)
        threading.Thread(target=task).start()

    def _get_anime_list_task(self):
        """Fetch the user's anime list and update cache, ensuring thread safety."""
        try:
            with self.lock:  # Ensure thread safety
                self.set_cursor_loading()  # Indicate loading
                self.anime_list = self.get_anime_list_cache()
                if not self.anime_list:  # Cache miss, fetch from API
                    self.anime_list = self.get_anime_list_api()
                    if self.anime_list and self.anime_list != -2:  # Successful API call
                        self.save_anime_list_cache()  # Update cache
                        self.user_anime_ids = {
                            anime["node"]["id"] for anime in self.anime_list if anime["list_status"]["status"] == "completed"
                        }
                        return True
                    else:
                        print("Failed to fetch anime list.")
                        return False
                else:  # Cache hit
                    self.user_anime_ids = {
                        anime["node"]["id"] for anime in self.anime_list if anime["list_status"]["status"] == "completed"
                    }
                    print("Loaded anime list from cache.")
                    return True
        except Exception as e:
            print(f"An error occurred: {e}")
            self.reset_cursor()
            return False

    def get_anime_list_api(self, status=None, fields="list_status,mean,rank,popularity,genres,num_episodes,start_season,studios"):
        """
        Fetch the user's anime list from the MyAnimeList API with support for pagination.
        
        Args:
            status (str): Filter by anime status (e.g., watching, completed, etc.).
            fields (str): Specific fields to fetch from the API.
            
        Returns:
            list or int: A list of anime data or -2 if the API call times out.
        """
        if not hasattr(self, 'username') or not self.username:
            messagebox.showwarning("Error", "Username is not set.")
            return

        base_url = f"https://api.myanimelist.net/v2/users/{self.username}/animelist"
        headers = {"X-MAL-CLIENT-ID": self.CLIENT_ID}
        params = {
            "fields": fields, # Can include list_status,alternative_titles,start_date,end_date,synopsis,mean,rank,popularity,num_list_users,num_scoring_users,nsfw,created_at,updated_at,media_type,status,genres,num_episodes,start_season,broadcast,source,average_episode_duration,rating,studios.
            "limit": 100,
            "nsfw": "true" # watching/completed/on_hold/dropped/plan_to_watch
        }
        if status:
            params["status"] = status

        all_data = []  # Accumulate all fetched data
        next_url = base_url

        try:
            while next_url:
                if next_url == base_url:  # Initial request
                    response = requests.get(next_url, headers=headers, params=params, timeout=10)
                else:  # Follow pagination
                    response = requests.get(next_url, headers=headers, timeout=10)

                response.raise_for_status()
                if "application/json" in response.headers.get("Content-Type", ""):
                    data = response.json()
                    all_data.extend(data.get("data", []))
                    next_url = data.get("paging", {}).get("next")
                else:
                    messagebox.showwarning("Error", "Unexpected content type in API response.")
                    break
        except requests.exceptions.Timeout:
            messagebox.showerror("Error", "API request timed out. Try again later.")
            return -2
        except requests.exceptions.RequestException as e:
            messagebox.showerror("Error", f"Request failed: {e}")
            return
        except json.JSONDecodeError:
            messagebox.showerror("Error", "Failed to decode JSON response.")
            return

        print(f"Success, fetched {len(all_data)} anime for user '{self.username}'.")
        return all_data

    def get_anime_list_api_threaded(self, callback=None, status=None, fields="list_status,mean,rank,popularity,genres,num_episodes,start_season,studios"):
        """
        Fetch the anime list in a separate thread and optionally call a callback on success.

        Args:
            callback (function, optional): Function to execute after the anime list is fetched.
            status (str, optional): Status filter for anime (e.g., "completed").
            fields (str, optional): Fields to fetch from the API.
        """
        def task():
            anime_list = self.get_anime_list_api(status=status, fields=fields)
            if anime_list and callback:
                # Execute the callback on the main thread
                self.root.after(0, lambda: callback(anime_list))

        threading.Thread(target=task).start()

    def get_anime_list_cache(self):
        """
        Load the user's anime list from the cache, if available and valid.
        
        Returns:
            list or None: The cached anime list, or None if no valid cache is found.
        """
        if not os.path.exists(self.ANIME_LIST_CACHE):
            return None

        try:
            with open(self.ANIME_LIST_CACHE, "r", encoding="utf-8") as cache_file:
                cache_data = json.load(cache_file)

            if self.username in cache_data:
                user_data = cache_data[self.username]
                cache_timestamp = datetime.fromisoformat(user_data.get("timestamp"))

                if datetime.now() - cache_timestamp < timedelta(hours=self.CACHE_EXPIRY_HOURS):
                    print(f"Using cached anime list for user: {self.username}.")
                    return user_data["anime_list"]
                else:
                    print(f"Cache expired for user: {self.username}.")
                    return None
            else:
                print(f"No cache found for user: {self.username}.")
                return None
        except Exception as e:
            messagebox.showerror("Error", f"Error loading cache: {e}")
            return None

    def save_anime_list_cache(self):
        """
        Save or update the user's anime list in the cache.
        """
        try:
            if os.path.exists(self.ANIME_LIST_CACHE):
                with open(self.ANIME_LIST_CACHE, "r", encoding="utf-8") as cache_file:
                    cache_data = json.load(cache_file)
            else:
                cache_data = {}

            cache_data[self.username] = {
                "timestamp": datetime.now().isoformat(),
                "anime_list": self.anime_list
            }

            with open(self.ANIME_LIST_CACHE, "w", encoding="utf-8") as cache_file:
                json.dump(cache_data, cache_file, indent=4, ensure_ascii=False)

            print(f"Cache saved successfully for user: {self.username}.")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving cache: {e}")

    def remove_anime_list_cache(self):
        """
        Remove the user's anime list from the cache.
        """
        try:
            if not os.path.exists(self.ANIME_LIST_CACHE):
                messagebox.showinfo("Info", "No cache file found.")
                return

            with open(self.ANIME_LIST_CACHE, "r", encoding="utf-8") as cache_file:
                cache_data = json.load(cache_file)

            if self.username in cache_data:
                del cache_data[self.username]
                with open(self.ANIME_LIST_CACHE, "w", encoding="utf-8") as cache_file:
                    json.dump(cache_data, cache_file, indent=4, ensure_ascii=False)
                messagebox.showinfo("Success", f"Cache removed successfully for user: {self.username}.")
            else:
                messagebox.showinfo("Info", f"No cached data found for user: {self.username}.")
        except Exception as e:
            messagebox.showerror("Error", f"Error removing cache: {e}")

    def update_anime_list_cache(self):
        """
        Update the user's anime list in the cache by fetching the latest data.
        
        Returns:
            int: 1 if successful, 0 otherwise.
        """
        updated = self.get_anime_list_api()
        if updated and updated != -2:
            self.anime_list = updated
            self.save_anime_list_cache()
            messagebox.showinfo("Success", f"Cache updated successfully for user: {self.username}.")
            return 1
        return 0

    def clear_anime_list_cache(self):
        """
        Clear the entire anime list cache file.
        """
        try:
            if os.path.exists(self.ANIME_LIST_CACHE):
                os.remove(self.ANIME_LIST_CACHE)
                messagebox.showinfo("Success", "Anime list cache file deleted successfully.")
            else:
                messagebox.showinfo("Info", "No anime list cache file found.")
        except Exception as e:
            messagebox.showerror("Error", f"Error deleting anime list cache file: {e}")

    def get_anime_details(self):
        """
        Fetch details for all anime in the user's list, leveraging caching and concurrency.
        """
        # Load cached anime details
        self.get_anime_details_cache()

        def fetch_details(anime):
            """
            Fetch details for a single anime, using cache if available.
            
            Args:
                anime (dict): Anime data from the list.
            
            Returns:
                int or None: Status code or None if an error occurs.
            """
            anime_id = str(anime["node"]["id"])
            anime_title = anime["node"]["title"]
            
            if anime_id in self.anime_details:
                return -2
            else:
                return self.get_anime_details_api(anime_id)

        # Use ThreadPoolExecutor to fetch details concurrently
        with ThreadPoolExecutor(max_workers=500) as executor: # No other way based on someone that seems like admin or smth https://myanimelist.net/forum/?topicid=2105546
            future_to_anime = {executor.submit(fetch_details, anime): anime for anime in self.anime_list} # https://myanimelist.net/forum/?topicid=2142532 no documentation about rate limits, From tests - 1/s limit at 140, 2/s limit at 180, without sleep limit at 180, with concurrent.futures can do atleast 1000 in 10s, but lockout after 10s for 3-10min
            try:
                for future in as_completed(future_to_anime):
                    result = future.result()
                    if result is None:  # Stop on error
                        messagebox.showerror(
                            "Error",
                            "A fetch returned None (indicating an error). Stopping updates and saving cache."
                        )
                        executor.shutdown(wait=False)
                        self.keep_on_top()
                        break
                    elif result == -1:  # Handle rate-limiting
                        messagebox.showerror(
                            "Error",
                            "Timeout or rate limiter hit. Wait 5-10 minutes and try again."
                        )
                        executor.shutdown(wait=False)
                        self.keep_on_top()
                        break
            except Exception as e:
                messagebox.showerror("Error", f"Unhandled exception: {e}")
                executor.shutdown(wait=False)
                self.keep_on_top()

        # Save updated details cache
        self.save_anime_details_cache()

    def get_anime_details_api(self, anime_id, fields="id,title,main_picture,,mean,rank,popularity,genres,num_episodes,start_season,studios,recommendations"):
        """
        Fetch detailed information about a specific anime from the MyAnimeList API.

        Args:
            anime_id (str): ID of the anime to fetch details for.
            fields (str): Fields to retrieve from the API.

        Returns:
            int or None: 1 for success, -1 for timeout, or None for failure.
        """
        base_url = f"https://api.myanimelist.net/v2/anime/{anime_id}"
        headers = {"X-MAL-CLIENT-ID": self.CLIENT_ID}
        params = {"fields": fields} # Can be id,title,main_picture,alternative_titles,start_date,end_date,synopsis,mean,rank,popularity,num_list_users,num_scoring_users,nsfw,created_at,updated_at,media_type,status,genres,my_list_status,num_episodes,start_season,broadcast,source,average_episode_duration,rating,pictures,background,related_anime,related_manga,recommendations,studios,statistics

        try:
            response = requests.get(base_url, headers=headers, params=params, timeout=10)
            response.raise_for_status()

            if "application/json" in response.headers.get("Content-Type", ""):
                self.anime_details[anime_id] = {
                    "timestamp": datetime.now().isoformat(),
                    "details": response.json()
                }
                return 1
            else:
                print(f"Unexpected content type: {response.headers.get('Content-Type')}")
                return None
        except requests.exceptions.Timeout:
            print(f"Request for Anime ID {anime_id} timed out.")
            return -1
        except requests.exceptions.RequestException as e:
            print(f"Request failed for Anime ID {anime_id}: {e}")
            return None
        except json.JSONDecodeError:
            print(f"Failed to decode JSON for Anime ID {anime_id}: {response.text}")
            return None

    def get_anime_details_cache(self):
        """
        Load cached anime details from the local cache file.
        """
        try:
            with open(self.ANIME_DETAILS_CACHE_FILE, 'r', encoding='utf-8') as file:
                self.anime_details = json.load(file)
                print("Anime details successfully loaded from cache.")
        except FileNotFoundError:
            print(f"Error: The file {self.ANIME_DETAILS_CACHE_FILE} was not found.")
        except json.JSONDecodeError:
            print("Error: The JSON file is not properly formatted.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def save_anime_details_cache(self):
        """
        Save updated anime details to the local cache file.
        """
        try:
            with open(self.ANIME_DETAILS_CACHE_FILE, "w", encoding="utf-8") as cache_file:
                json.dump(self.anime_details, cache_file, indent=4, ensure_ascii=False)
            print("Details cache saved successfully")
        except Exception as e:
            print(f"Error saving details cache: {e}")

    def single_update_anime_details_cache(self, anime_id):
        """
        Update details for a single anime in the cache.

        Args:
            anime_id (str): ID of the anime to update.
        """
        self.get_anime_details_cache()
        self.get_anime_details_api(anime_id)
        self.save_anime_details_cache()

    def update_anime_details_cache(self):
        """
        Update details for all anime in the user's list, using caching and concurrency.
        """
        self.get_anime_details_cache()

        def fetch_and_update(anime):
            """
            Fetch and update details for a single anime.
            
            Args:
                anime (dict): Anime data from the list.
            
            Returns:
                int or None: Status code or None if an error occurs.
            """
            anime_id = str(anime["node"]["id"])
            return self.get_anime_details_api(anime_id)

        all_updates_successful = True

        with ThreadPoolExecutor(max_workers=500) as executor:
            future_to_anime = {executor.submit(fetch_and_update, anime): anime for anime in self.anime_list}
            try:
                for future in as_completed(future_to_anime):
                    result = future.result()
                    if result is None:
                        messagebox.showerror(
                            "Error",
                            "A fetch returned None (indicating an error). Stopping updates and saving cache."
                        )
                        executor.shutdown(wait=False)
                        all_updates_successful = False
                        break
                    elif result == -1:
                        messagebox.showerror(
                            "Error",
                            "Timeout or rate limiter hit. Wait 5-10 minutes and try again."
                        )
                        executor.shutdown(wait=False)
                        all_updates_successful = False
                        break
            except Exception as e:
                messagebox.showerror("Error", f"Unhandled exception: {e}")
                executor.shutdown(wait=False)
                all_updates_successful = False

        self.save_anime_details_cache()

        if all_updates_successful:
            messagebox.showinfo("Success", f"All anime details updated successfully for user {self.username}.")
        else:
            print("There was an error; not all details were fetched.")

    def clear_anime_details_cache(self):
        """
        Clear the anime details cache by deleting the cache file.
        """
        try:
            if os.path.exists(self.ANIME_DETAILS_CACHE_FILE):
                os.remove(self.ANIME_DETAILS_CACHE_FILE)
                messagebox.showinfo("Success", "Anime details cache file deleted successfully.")
            else:
                messagebox.showinfo("Info", "No anime details cache file found to delete.")
        except Exception as e:
            messagebox.showerror("Error", f"Error deleting anime details cache file: {e}")

    def update_all_cache_threaded(self, callback=None):
        """
        Start a thread to update all caches and optionally call a callback when done.
        """
        threading.Thread(target=self._update_all_cache_task, args=(callback,)).start()

    def _update_all_cache_task(self, callback=None):
        """
        Task to update all caches in a thread-safe manner, without blocking the GUI.
        Optionally calls a callback function upon completion.
        """
        try:
            window = self.second_window  # Reference to the secondary window
            self.set_cursor_loading(window)  # Set cursor to loading
            with self.lock:
                self.update_all_cache()  # Perform the cache update
            print("Cache updated successfully.")
            if callback:
                self.root.after(0, callback)  # Schedule the callback on the main thread
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            self.reset_cursor(window)  # Reset cursor to normal

    def update_all_cache(self):
        """
        Update both the anime list and anime details caches.
        """
        if self.update_anime_list_cache():
            self.update_anime_details_cache()  # Update details only if list update succeeds
        self.keep_on_top()  # Ensure the current window stays on top

    def clear_all_cache(self):
        """
        Clear both the anime list cache and the anime details cache.
        """
        self.clear_anime_list_cache()
        self.clear_anime_details_cache()
        self.clear_ranked_anime_cache()
        self.keep_on_top()

    def showcase_list_threaded(self):
        """
        Start a thread to showcase the user's anime list without blocking the GUI.
        """
        threading.Thread(target=self._showcase_list_task).start()

    def _showcase_list_task(self):
        """
        Task to showcase the user's anime list, executed in a separate thread.
        """
        try:
            window = self.second_window  # Reference to the secondary window
            self.set_cursor_loading(window)  # Set cursor to loading
            self.showcase_list()
            print("Showcase created.")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            self.reset_cursor(window)  # Reset cursor to normal

    def showcase_list(self):
        """
        Display the user's anime list in a new scrollable window with thumbnails and details.
        """
        anime_list = self.anime_list if self.anime_list else self.get_anime_list_cache()

        if not anime_list:
            messagebox.showinfo("Info", f"No anime list found for user: {self.username}.")
            return

        # Create a scrollable window to display the anime list
        showcase_window = tk.Toplevel(self.root)
        showcase_window.title(f"{self.username}'s Anime List")
        showcase_window.geometry("800x600")
        canvas = tk.Canvas(showcase_window)
        scrollbar = tk.Scrollbar(showcase_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        # Update the canvas scrolling region dynamically
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bind mouse wheel scrolling to the canvas
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-1 * (e.delta // 120), "units"))

        # Store fetched images in a dictionary
        images = {}
        with ThreadPoolExecutor(max_workers=20) as executor:
            for anime, future in zip(anime_list, executor.map(self.fetch_image, anime_list)):
                images[anime['node']['id']] = future

        # Populate the scrollable frame with anime details
        for anime in anime_list:
            frame = tk.Frame(scrollable_frame, pady=5)
            frame.pack(fill="x", padx=10)

            # Display the thumbnail image if available
            img = images.get(anime['node']['id'])
            if img:
                img_tk = ImageTk.PhotoImage(img)
                tk.Label(frame, image=img_tk).pack(side="left", padx=5)
                frame.image = img_tk  # Prevent image from being garbage collected

            # Display the anime's details
            details = (
                f"Title: {anime['node']['title']}\n"
                f"Genres: {', '.join(g['name'] for g in anime['node'].get('genres', []))}\n"
                f"Episodes: {anime['node'].get('num_episodes', 'Unknown')} | "
                f"Status: {anime['list_status']['status']} | "
                f"Score: {anime['list_status']['score']}"
            )
            tk.Label(
                frame, text=details, font=("Arial", 10), anchor="w", justify="left", wraplength=600
            ).pack(fill="x")

        def update_showcase():
            """
            Trigger update process with cursor indication and refresh the showcase window after updates complete.
            """
            def on_update_complete(anime_list):
                # Callback to execute after the update is complete
                if anime_list!= -2:
                    self.anime_list = anime_list  # Update the anime list with the new data
                self.reset_cursor(showcase_window)
                showcase_window.destroy()
                self.showcase_list_threaded()  # Recreate the showcase window

            def update_task():
                # Set cursor to loading
                self.set_cursor_loading(showcase_window)
                # Perform the update asynchronously using get_anime_list_threaded
                self.get_anime_list_api_threaded(callback=on_update_complete)

            # Start the update task in a new thread
            threading.Thread(target=update_task).start()

        update_button = tk.Button(showcase_window, text="Reload List", command=update_showcase)
        update_button.pack(pady=10, side="bottom")

    def create_recommendations_threaded(self):
        """
        Start a thread to create recommendations and display them without blocking the GUI.
        """
        threading.Thread(target=self._create_recommendations_task).start()

    def _create_recommendations_task(self):
        """
        Task to generate recommendations and display the list in a thread-safe manner.
        """
        try:
            window = self.second_window  # Reference to the secondary window
            self.set_cursor_loading(window)  # Set cursor to loading
            with self.lock:
                self.create_recommendations_list()  # Display the recommendations
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            self.reset_cursor(window)  # Reset cursor to normal

    def create_recommendations(self, filter=True):
        """
        Generate anime recommendations based on the user's completed anime list.

        Args:
            filter (bool): If True, exclude recommendations already in the user's anime list.
        """
        self.recommendation_list = {}

        # Iterate through the user's completed anime list
        for anime in self.anime_list:
            if anime["list_status"]["status"] == "completed":
                anime_id = str(anime["node"]["id"])
                anime_detail = self.anime_details[anime_id]
                user_score = anime["list_status"]["score"]  # User's score for the anime

                for recommendation in anime_detail["details"]["recommendations"]:
                    recommended_anime = recommendation["node"]
                    recommended_id = recommended_anime["id"]

                    original_num_recommendations = recommendation["num_recommendations"]
                    modified_recommendations = original_num_recommendations * user_score

                    if filter and recommended_id in self.user_anime_ids:
                        if recommended_id in self.recommendation_list_train:
                            # Accumulate recommendations if already exists
                            self.recommendation_list_train[recommended_id]["num_recommendations"] += modified_recommendations
                        else:
                            # Add new recommendation
                            self.recommendation_list_train[recommended_id] = {
                                "node": recommended_anime,
                                "num_recommendations": modified_recommendations,
                            }
                        continue

                    if recommended_id in self.recommendation_list:
                        # Accumulate recommendations if already exists
                        self.recommendation_list[recommended_id]["num_recommendations"] += modified_recommendations
                    else:
                        # Add new recommendation
                        self.recommendation_list[recommended_id] = {
                            "node": recommended_anime,
                            "num_recommendations": modified_recommendations,
                        }

        # Sort recommendations by adjusted recommendation count
        self.sorted_recommendations = sorted(
            self.recommendation_list.values(),
            key=lambda x: x["num_recommendations"],
            reverse=True
        )
        self.add_recommended_anime()

    def create_recommendations_list(self):
        """
        Open a new window to showcase the top 100 recommendations with an option to update.
        """
        # Ensure recommendations are available
        if not hasattr(self, 'sorted_recommendations') or not self.sorted_recommendations:
            messagebox.showinfo("Info", "No recommendations found. Please generate recommendations first.")
            return

        # Limit to the top 100 recommendations
        top_recommendations = self.sorted_recommendations[:100]

        # Create a scrollable window to display recommendations
        recommendations_window = tk.Toplevel(self.root)
        recommendations_window.title(f"{self.username}'s Top Recommendations")
        recommendations_window.geometry("800x600")

        # Frame for canvas and scrollbar
        container = tk.Frame(recommendations_window)
        container.pack(fill="both", expand=True)

        canvas = tk.Canvas(container)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        # Bind canvas to dynamic resizing
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Store fetched images in a dictionary
        images = {}
        with ThreadPoolExecutor(max_workers=20) as executor:
            for recommendation, future in zip(top_recommendations, executor.map(self.fetch_image, top_recommendations)):
                images[recommendation["node"]["id"]] = future

        # Populate the scrollable frame with recommendations
        for recommendation in top_recommendations:
            frame = tk.Frame(scrollable_frame, pady=5)
            frame.pack(fill="x", padx=10)

            anime = recommendation["node"]
            img = images.get(anime["id"])

            if img:
                img_tk = ImageTk.PhotoImage(img)
                tk.Label(frame, image=img_tk).pack(side="left", padx=5)
                frame.image = img_tk  # Reference to avoid garbage collection

            details = f"Title: {anime['title']}\nPoints: {recommendation['num_recommendations']}"
            tk.Label(frame, text=details, font=("Arial", 10), anchor="w", justify="left", wraplength=600).pack(fill="x")

        # Enable mouse wheel scrolling
        def on_mouse_wheel(event):
            canvas.yview_scroll(-1 * int(event.delta / 120), "units")

        recommendations_window.bind_all("<MouseWheel>", on_mouse_wheel)

        # Add Update Recommendations Button
        def update_recommendations():
            """
            Trigger update process with cursor indication and refresh the recommendations window after updates complete.
            """
            def on_update_complete():
                # Callback to execute after the update is complete
                self.reset_cursor(recommendations_window)
                recommendations_window.destroy()
                self.create_recommendations_threaded()

            def update_task():
                # Set cursor to loading
                self.set_cursor_loading(recommendations_window)
                # Perform the update asynchronously
                self.update_all_cache_threaded(callback=on_update_complete)

            # Start the update task in a new thread
            threading.Thread(target=update_task).start()


        update_button = tk.Button(recommendations_window, text="Update Recommendations", command=update_recommendations)
        update_button.pack(pady=10, side="bottom")

    def add_recommended_anime(self):
        """
        Add all recommended anime to anime_details_cache if missing.
        """
        # Load cached anime details
        def fetch_details(anime):
            """
            Fetch details for a single anime, using cache if available.
            
            Args:
                anime (dict): Anime data from the list.
            
            Returns:
                int or None: Status code or None if an error occurs.
            """
            anime_id = str(anime["node"]["id"])
            
            if anime_id in self.anime_details:
                return -2
            else:
                return self.get_anime_details_api(anime_id)

        # Use ThreadPoolExecutor to fetch details concurrently
        with ThreadPoolExecutor(max_workers=500) as executor: # No other way based on someone that seems like admin or smth https://myanimelist.net/forum/?topicid=2105546
            future_to_anime = {executor.submit(fetch_details, anime): anime for anime in self.sorted_recommendations[:200]} # https://myanimelist.net/forum/?topicid=2142532 no documentation about rate limits, From tests - 1/s limit at 140, 2/s limit at 180, without sleep limit at 180, with concurrent.futures can do atleast 1000 in 10s, but lockout after 10s for 3-10min
            try:
                for future in as_completed(future_to_anime):
                    result = future.result()
                    if result is None:  # Stop on error
                        messagebox.showerror(
                            "Error",
                            "A fetch returned None (indicating an error). Stopping updates and saving cache."
                        )
                        executor.shutdown(wait=False)
                        self.keep_on_top()
                        break
                    elif result == -1:  # Handle rate-limiting
                        messagebox.showerror(
                            "Error",
                            "Timeout or rate limiter hit. Wait 5-10 minutes and try again."
                        )
                        executor.shutdown(wait=False)
                        self.keep_on_top()
                        break
            except Exception as e:
                messagebox.showerror("Error", f"Unhandled exception: {e}")
                executor.shutdown(wait=False)
                self.keep_on_top()

        # Save updated details cache
        self.save_anime_details_cache()
        
    def fetch_image(self, anime):
        """
        Fetch the thumbnail image for an anime.

        Args:
            anime (dict): Anime data from the list.

        Returns:
            PIL.Image or None: Resized image or None if fetching fails.
        """
        try:
            url = anime['node']['main_picture']['medium']
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content))
        except requests.RequestException:
            return None
            
    def set_cursor_loading(self, window=None):
        """
        Set the mouse pointer to a loading (busy) state.

        Args:
            window (tk.Toplevel or tk.Tk, optional): The target window to update. Defaults to the root window.
        """
        target = window or self.root
        target.config(cursor="watch")
        target.update_idletasks()

    def reset_cursor(self, window=None):
        """
        Reset the mouse pointer to the default state.

        Args:
            window (tk.Toplevel or tk.Tk, optional): The target window to update. Defaults to the root window.
        """
        target = window or self.root
        target.config(cursor="")
        target.update_idletasks()

    def showcase_ml_recommendations(self):
        """
        Display the ML-based anime recommendations in a scrollable window,
        handling both anime from details cache and from ranking list.
        """
        # Get ranking data for reference (we'll use this to get details for anime not in cache)
        flattened_ranking_list = []
        if hasattr(self, 'ranking_list') and self.ranking_list:
            for batch in self.ranking_list:
                flattened_ranking_list.extend(batch)
            
        # Create a lookup dictionary from ranking list for easy access
        ranking_lookup = {}
        for item in flattened_ranking_list:
            anime_id = str(item["node"]["id"])
            ranking_lookup[anime_id] = item
                
        # Get ML recommendations (top 50)
        recommendations = self.ml_recommender.get_ml_recommendations(top_n=50)
        
        print(f"Total ML recommendations: {len(recommendations)}")
        
        if not recommendations:
            messagebox.showinfo("Info", "No ML recommendations available.")
            return

        # Create a scrollable window to display recommendations
        showcase_window = tk.Toplevel(self.root)
        showcase_window.title(f"ML Recommendations for {self.username}")
        showcase_window.geometry("800x600")
        canvas = tk.Canvas(showcase_window)
        scrollbar = tk.Scrollbar(showcase_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        # Update the canvas scrolling region dynamically
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bind mouse wheel scrolling to the canvas
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-1 * (e.delta // 120), "units"))

        # Store fetched images in a dictionary
        images = {}
        
        # Prepare anime data for image fetching
        anime_data = []
        for anime_id, score, title in recommendations:
            # Try to get data from anime_details first
            if anime_id in self.anime_details:
                details = self.anime_details[anime_id]["details"]
                node_data = {
                    'node': {
                        'id': details['id'],
                        'title': details['title'],
                        'main_picture': details['main_picture']
                    }
                }
                anime_data.append((anime_id, node_data))
            # If not in anime_details, try to get from ranking list
            elif anime_id in ranking_lookup:
                anime_data.append((anime_id, ranking_lookup[anime_id]))
        
        # Fetch images concurrently
        with ThreadPoolExecutor(max_workers=20) as executor:
            for anime_id, node_data in anime_data:
                try:
                    images[anime_id] = executor.submit(self.fetch_image, node_data).result()
                except Exception as e:
                    print(f"Error fetching image for anime {anime_id}: {e}")

        # Counter for actual displayed recommendations
        displayed_count = 0
        
        # Populate the scrollable frame with recommendations
        for anime_id, predicted_score, title in recommendations:
            # Track if we have details to display
            has_details = False
            details_source = None
            
            # Try to get from anime_details first
            if anime_id in self.anime_details:
                details = self.anime_details[anime_id]["details"]
                has_details = True
                details_source = "cache"
            # If not in anime_details, try to get from ranking list
            elif anime_id in ranking_lookup:
                details = ranking_lookup[anime_id]["node"]
                has_details = True
                details_source = "ranking"
            
            # Skip if we couldn't find details
            if not has_details:
                print(f"Skipping anime ID {anime_id} - no details available")
                continue
                
            # Create frame for this anime
            frame = tk.Frame(scrollable_frame, pady=5)
            frame.pack(fill="x", padx=10)

            # Display the thumbnail image if available
            img = images.get(anime_id)
            if img:
                img_tk = ImageTk.PhotoImage(img)
                tk.Label(frame, image=img_tk).pack(side="left", padx=5)
                frame.image = img_tk  # Prevent image from being garbage collected

            # Display the anime's details including the ML predicted score
            details_text = (
                f"Title: {title}\n"
                f"Predicted Score: {predicted_score}\n"
                f"MAL Score: {details.get('mean', 'N/A')} | "
                f"Rank: {details.get('rank', 'N/A')} | "
                f"Episodes: {details.get('num_episodes', 'N/A')}\n"
                f"Genres: {', '.join(g['name'] for g in details.get('genres', []))}"
            )
            tk.Label(
                frame, text=details_text, font=("Arial", 10), anchor="w", justify="left", wraplength=600
            ).pack(fill="x")
            
            displayed_count += 1

        print(f"Successfully displayed {displayed_count} out of {len(recommendations)} recommendations")
        
        # Add a button to close the window
        close_button = tk.Button(showcase_window, text="Close", command=showcase_window.destroy)
        close_button.pack(pady=10, side="bottom")
        
    def create_ML_recommendations(self):
        """
        Generate anime recommendations based on the user's completed anime list using ML.
        """
        # Train ML
        print("Starting ML training...")
        success = self.ml_recommender.tune_and_train()
        if success:
            print("Training completed successfully!")
        else:
            print("Training failed.")
        # Get recommendations
        print("Generating recommendations...")
        self.showcase_ml_recommendations()
        print("Recommendations completed successfully!")

    def create_ML_recommendations_threaded(self):
        """
        Start a thread to create ml recommendations and display them without blocking the GUI.
        """
        threading.Thread(target=self._create_ML_recommendations_task).start()
    
    def _create_ML_recommendations_task(self):
        """
        Task to generate ml recommendations and display the list in a thread-safe manner.
        """
        try:
            window = self.second_window  # Reference to the secondary window
            self.set_cursor_loading(window)  # Set cursor to loading
            with self.lock:
                self.create_ML_recommendations()  # Display the recommendations
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            self.reset_cursor(window)  # Reset cursor to normal

    def get_ranked_anime(self):
        data = self.get_ranked_anime_cache()
        if(data):
            self.ranking_list = data
        else:
            self.get_ranked_anime_api()

    def get_ranked_anime_api_threaded(self):
        """
        Start a thread to update ranked anime list without blocking the GUI.
        """
        threading.Thread(target=self._get_ranked_anime_api_task).start()

    def _get_ranked_anime_api_task(self):
        """
        Task to update ranked anime list, executed in a separate thread.
        """
        try:
            window = self.second_window  # Reference to the secondary window
            self.set_cursor_loading(window)  # Set cursor to loading
            with self.lock:
                self.get_ranked_anime_api()
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            self.reset_cursor(window)  # Reset cursor to normal
        
    def get_ranked_anime_api(self):
        """
        Fetch anime rankings from the MyAnimeList API for first 2500 anime.
        """
        self.ranking_list = []
        offset=0
        while offset < 2501:
            self.ranking_list.append(self.get_anime_rankings("all", 500, offset,"id,title,main_picture,,mean,rank,popularity,genres,num_episodes,start_season,studios"))
            offset += 500
        self.save_ranked_anime()

    def get_ranked_anime_cache(self):
        """
        Load the global ranking list from the cache, if available.
        
        """
        if not os.path.exists(self.ANIME_GLOBAL_RANKINGS_FILE):
            return None

        try:
            with open(self.ANIME_GLOBAL_RANKINGS_FILE, "r", encoding="utf-8") as cache_file:
                cache_data = json.load(cache_file)
            if cache_data:
                print(f"Using cached ranking list.")
                return cache_data
            else:
                print("Ranking cache empty.")
                return None
        except Exception as e:
            messagebox.showerror("Error", f"Error loading cache: {e}")
            return None
    
    def save_ranked_anime(self):
        try:
            with open(self.ANIME_GLOBAL_RANKINGS_FILE, "w", encoding="utf-8") as file:
                json.dump(self.ranking_list, file, indent=4, ensure_ascii=False)
            print(f"Data saved to {self.ANIME_GLOBAL_RANKINGS_FILE}")
        except Exception as e:
            print(f"An error occurred while saving the file: {e}")

    def clear_ranked_anime_cache(self):
        """
        Clear the anime details cache by deleting the cache file.
        """
        try:
            if os.path.exists(self.ANIME_GLOBAL_RANKINGS_FILE):
                os.remove(self.ANIME_GLOBAL_RANKINGS_FILE)
                messagebox.showinfo("Success", "Anime details cache file deleted successfully.")
            else:
                messagebox.showinfo("Info", "No anime details cache file found to delete.")
        except Exception as e:
            messagebox.showerror("Error", f"Error deleting anime details cache file: {e}")

    def test(self):
        self.ml_recommender.tune_and_train()
        results = self.ml_recommender.get_ml_recommendations(top_n=250)
        print(len(results))
        print(results)

# Main application entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = MyAnimeListApp(root)
    root.mainloop()
