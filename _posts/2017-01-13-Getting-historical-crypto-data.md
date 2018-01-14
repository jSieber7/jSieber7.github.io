# Tutorial: How to easily get cryptocurrency data in Python
*An accompanying Jupyter Notebook will be added a later date*

While there are many existing APIs to find live cryptocurrency data, there seems to be a lack of APIs that allow a user to download the historcial price and volume of a given coin. Fortunately, a website called https://coinmarketcap.com/ gives us the historical information of over 1000 coins in a neat table. The question is: how do we get this table onto our own computers in an easily readable way for the coins and type periods we want?

<img src="https://i.imgur.com/RoIsCvV.png" title="Coinmarketcap.com interface" />








## Introducting Beautiful Soup

Beautiful Soup allows users to easily parse html websites and return the specific parts that they want. For example, we will use Beautiful Soup to get all of the elements with the 'tr' tags in order to get read only the table's cells on. We find out which elements have which tags through the console window. In addition to the beautiful soup library, we will need urllib.request to go to the webpage, pandas to store our data in a dataframe, and xlwt to output our data into an excel spreadsheet (optional). We will only be using the basic functionalities of these libraries. We will first start with getting a years worth of Bitcoin data. Now lets start coding!

<img src="https://i.imgur.com/27Ei4MZ.jpg?1" title="Using the console in chrome" />

```
# Importing our libraries for web parsing
import bs4 as bs
import urllib.request
import pandas as pd
import xlwt

# Visiting the website through Python
url = 'https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20170114&end=20180114'
link = urrlib.request.urlopen(url).read()

# Activating beautiful soup and getting all of the text with the tags <tr>
soup = bs.BeautifulSoup(link, "html.parser")
prices_table = soup.find_all('tr')

# Now we make a list of dictionaries to create a dataframe.
# Notice how we loop over the prices_table variable that beautiful soup created
list_of_dic = []
for count, itemset in enumerate(prices_table):
    date = itemset.text.splitlines()[1]
    open = itemset.text.splitlines()[2]
    high = itemset.text.splitlines()[3]
    low = itemset.text.splitlines()[4]
    close = itemset.text.splitlines()[5]
    volume = itemset.text.splitlines()[6]
    marketcap = itemset.text.splitlines()[7]
    dictionary = {'Date' : date, 'Coin' : coin.capitalize(), 'Opening Price' : open,'Closing Price' : close, 'Low' :
        low, 'High': high, 'Volume' : volume, 'Market Cap': marketcap}
    list_of_dic.append(dictionary)

# Now we simply put this dictionary into a dataframe
df = pd.DataFrame(list_of_dic)

# We drop the first column because we already made a header column
df = df.drop(df.index[0])

# We turn our date column into the time of date for easily manipulation
df['Date'] = pd.to_datetime(df['Date'])
```
## See how easy that was?

You may be asking yourself, how do we do more than one coin at a time, or how do we change the dates without looking up the url everytime? Well, it is actually pretty simple to do through python functions. Check out the future follow-up post for more details.
