\# Yelp Dataset Field Glossary



\## Review Fields (review\_subset.json)

\- `review\_id`: Unique identifier for this review

\- `user\_id`: The user who wrote the review

\- `item\_id`: The business being reviewed

\- `stars`: Rating given (1.0 = very bad, 5.0 = excellent)

\- `useful`: Number of users who marked this review as useful

\- `funny`: Number of users who marked this review as funny

\- `cool`: Number of users who marked this review as cool

\- `text`: The actual written review text

\- `date`: Date the review was posted



\## User Fields (user\_subset.json)

\- `user\_id`: Unique user identifier

\- `name`: User display name

\- `review\_count`: Total reviews written — higher means more experienced

\- `yelping\_since`: Date joined Yelp — longer means more reliable

\- `useful/funny/cool`: Total votes received across all their reviews

\- `elite`: Years awarded Elite status — Elite users are highly trusted

\- `friends`: List of friend user\_ids

\- `fans`: Number of followers

\- `average\_stars`: User's historical average rating — KEY FIELD: reveals if they rate high or low

\- `compliment\_hot`: Compliments for being popular

\- `compliment\_writer`: Compliments for good writing quality

\- `compliment\_funny`: Compliments for funny reviews

\- `compliment\_cool`: Compliments for cool reviews



\## Business/Item Fields (item\_subset.json)

\- `item\_id`: Unique business identifier

\- `name`: Business name

\- `address/city/state/postal\_code`: Location info

\- `stars`: Overall average star rating (1.0-5.0)

\- `review\_count`: Total reviews received

\- `is\_open`: 1 = open, 0 = permanently closed

\- `categories`: Business type (e.g. "Restaurants, Italian, Pizza")

\- `attributes`: Dictionary of features (see below)

\- `hours`: Opening hours per day



\## Business Attributes

\- `RestaurantsPriceRange2`: 1 (cheap) to 4 (expensive)

\- `NoiseLevel`: quiet / average / loud / very\_loud

\- `Alcohol`: full\_bar / beer\_and\_wine / none

\- `WiFi`: free / paid / no

\- `RestaurantsAttire`: casual / dressy / formal

\- `OutdoorSeating`: True/False

\- `RestaurantsReservations`: True/False

\- `RestaurantsTakeOut`: True/False

\- `RestaurantsDelivery`: True/False

\- `GoodForKids`: True/False

\- `Ambience`: romantic / intimate / classy / hipster / divey / touristy



\## Scoring Notes

\- Goal: predict `stars` (1.0-5.0) and `text` for a given user\_id + item\_id

\- Accuracy measured by MAE (Mean Absolute Error) — lower is better

\- user `average\_stars` is the strongest signal for predicting their rating

\- Always cross-reference user's past reviews of similar businesses

