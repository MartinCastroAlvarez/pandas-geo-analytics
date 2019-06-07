# Jaunt Challenge

## Corpus:
The input to your program is a text file containing a list of restaurants. The format for the file is:
Each column is separated by a comma (,) and each line represents a single restaurant. 
```
| Data          | Type          | Description                            |
| ------------- |-------------- | -------------------------------------- |
| Place         | String        | The name of the restaurant             |
| Address       | String        | The street address of the restaruant   |
| Latitude      | Double        | The latitue of the restaurant          |
| Longitude     | Double        | The longitude of the restaurant        |
| Tips          | String        | A brief description of the restaurant  |
```

## Goal:
The goal of the program is to parse a list of such restaurants and print out to the command line (stdout) answers to the following questions. A list of sample restaurants is sent as a separate file, so you can test your code with them. Assume each question has exactly one valid answer.

## Questions:
1. What is the number of unique restaurants present in the file?
2. Which two restaurants are furthest apart?  which two are closest?  What are the distances?
3. Which restaurants mention menu items in the `tips` section costing more than $10?
4. Classify each restaurant into one of the following two categories using any technique you prefer.
```
Category 1: Restaurants known for drinks
Category 2: Restaurants known for food
```

## Instructions:
Download the app from GitHub:
```
git clone https://github.com/MartinCastroAlvarez/jaunt.git
```
Install the application using these instructions:
```
cd jaunt
virtualenv -p python3 .env
source .env/bin/activate
pip install -r requirements.txt
```
Run the script. Checkout out all the script flags:
```
python3 jaunt.py --help
```
Returns:
```
usage: jaunt.py [-h] [--debug] [--no-debug] [--limit LIMIT] [--path PATH]

This method will be called by executing this script from the CLI. If script is
executed in debug mode, logger messages will be printed to STDOUT.

optional arguments:
  -h, --help            show this help message and exit
  --debug               Run script in debug mode. (default: False)
  --no-debug
  --limit LIMIT, -l LIMIT
                        Maximum amount of results to show. (default: 5)
  --path PATH, -p PATH  Datasource path. (default: ./restaurants.csv)
```

## Examples 1:
Run the script using the small dataset version:
```
python3 jaunt.py --path small.csv --limit 3
```
Returns:
```
------------------------------
1.1: Total Restaurants: 7
1.2: Unique Restaurants: 5
------------------------------
2.1: Furthest Restaurants:
               Place_To            Place_From   Distance
0  abita-brewingcompany             45-tchoup  62.972325
5  adams-street-grocery  abita-brewingcompany  60.012411
6            ale-on-oak  abita-brewingcompany  59.821664
------------------------------
2.2: Nearest Restaurants:
               Place_To            Place_From  Distance
9            ale-on-oak  adams-street-grocery  0.551171
2  adams-street-grocery             45-tchoup  3.817219
3            ale-on-oak             45-tchoup  4.306705
------------------------------
3: Expensive Restaurants:
             Place_Title  Tips_ExpensiveScore
3  Adams Street Grocery                  10.0
------------------------------
4.1: Restaurants by Food:
   Place_Title  Tips_FoodScore
4  Ale on Oak         4.658361
5  Ale on Oak         4.658361
6  Ale on Oak         4.658361
------------------------------
4.2: Restaurants by Drinks:
             Place_Title  Tips_DrinkScore
0             45 Tchoup               0.0
1  Abita BrewingCompany               0.0
2  Ace Hotel NewOrleans               0.0
------------------------------
```

## Examples 2:
Use the large dataset to extract results:
```
python3 jaunt.py --path restaurants.csv --limit 10
```
Returns:
```
------------------------------
1.1: Total Restaurants: 255
1.2: Unique Restaurants: 243
------------------------------
2.1: Furthest Restaurants:
                    Place_To                   Place_From    Distance
20432          rocky-carlo-s                     herbyk-s  459.001952
20414             panda-king                     herbyk-s  458.999414
20358  hong-kong-food-market                     herbyk-s  458.936710
16388               herbyk-s  dong-phuong-oriental-bakery  456.540095
16227               herbyk-s   dong-phuong-orientalbakery  456.535623
20404              nineroses                     herbyk-s  455.144799
20449     tan-dinhrestaurant                     herbyk-s  455.144443
4068                herbyk-s                    bar-redux  452.847864
20484       vaughan-s-lounge                     herbyk-s  452.721552
1302                herbyk-s    algiers-central-marketinc  452.437172
------------------------------
2.2: Nearest Restaurants:
                                   Place_To                                Place_From  Distance
19250                         gene-spo-boys                            gene-s-po-boys       0.0
26000                               oxlot-9                                    oxlot9       0.0
29382                           verti-marte                                vertimarte       0.0
27057                    ralph-s-on-thepark                       ralph-s-on-the-park       0.0
28434                        the-rusty-nail  sugar-bowl-extravaganza-at-the-rustynail       0.0
12549  the-wine-cellar-at-commander-spalace                        commander-s-palace       0.0
28997                         the-grillroom                            the-grill-room       0.0
28227                       sobourestaurant                          sobou-restaurant       0.0
27258            red-rooster-snowball-stand                 red-rooster-snowballstand       0.0
18377            frady-s-one-stop-foodstore               frady-s-one-stop-food-store       0.0
------------------------------
3: Expensive Restaurants:
                                           Place_Title  Tips_ExpensiveScore
26                                        Blue OakBBQ                 425.0
141                                          Mannings                 250.0
54          Chez Champagne Gosset Dinner atGalatoires                 225.0
95   Fireworks in the Mississippi in the SteamboatN...                175.0
63                                  Commander'sPalace                 100.0
105                      Gatsby Party atEffervescence                  80.0
125            Kermit Ruffins at The Little GemSaloon                  80.0
68                                            Criollo                  75.0
24                                 Beth Biundo Sweets                  69.0
196                                            Rue127                  65.0
------------------------------
4.1: Restaurants by Food:
                     Place_Title  Tips_FoodScore
74   Deanie's SeafoodRestaurant        16.236501
172                   PalaceCaf        13.897803
227               The GrillRoom        11.031991
55                   China Rose        10.906220
60                CochonButcher        10.208961
68                      Criollo        10.002220
124          Kartchner'sGrocery         9.343790
91               ElysianSeafood         8.825229
187          Rampart Food Store         8.715011
38          Bullet's Sports Bar         8.612483
------------------------------
4.2: Restaurants by Drinks:
                                   Place_Title  Tips_DrinkScore
214                             Ted's Frostop          2.351705
102                      Frey Smoked Meat Co.          1.652953
154                      Molly's at theMarket          1.344834
11                              Atomic Burger          1.041667
145                              Markey's Bar          0.931371
204                                     SoBou          0.831783
130                                LaProvence          0.807432
55                                 China Rose          0.779016
105              Gatsby Party atEffervescence          0.712495
210  Sugar Bowl Extravaganza at The RustyNail          0.555299
------------------------------
```
