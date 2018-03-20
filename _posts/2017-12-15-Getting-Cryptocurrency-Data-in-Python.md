*Learn how to get historical crypto data off the web.*
### *Jacob Sieber*
### *December 15, 2017 (Updated Feburary 27)*


<<<<<<< HEAD
# Part 1: Getting our data
=======
# Part 1: Getting our Data
>>>>>>> 7aace96aee4342c545db9e8001b128a59b683cd5
***
   When building a data science project, there comes times when we want to get data outside of what is readily available. Often times, we want data from sites like Twitter or Facebook. These sites provide there own API (programming interface) to easily get data from through languages such as R and python. When there are no APIs (or APIs don't provide enough data), we can obtain data through web scraping. Don't let web scraping intimidate you, through libraries such as Beautiful Soup, web scraping has never been easier. 

   While there are many existing APIs to find live cryptocurrency data, there seems to be a lack of APIs that allow a user to download the historcial price and volume of a given coin. Fortunately, a website called https://coinmarketcap.com/ gives us the historical information of over 1000 coins in a neat table. The question is: how do we get this table onto our own computers for the coins and type periods we want? 

  
<img src="https://i.imgur.com/RoIsCvV.png" title="Coinmarketcap.com interface" />
    

### Introducting Beautiful Soup
    
  Beautiful Soup allows users to easily parse html websites and return the specific parts that they want. For example, we will use Beautiful Soup to get all of the elements with the 'tr' tags in order to get read only the table cells. We find out which elements have which tags through the console window. In addition to the beautiful soup library, we will need urllib.request to get python to navigate to the webpage, pandas to store our data in a dataframe, and xlwt to output our data into an excel spreadsheet (optional). We will only be using the basic functionalities of these libraries. We will first start with getting a years worth of Bitcoin data. Now lets start coding!
  
  
  
  <img src="https://i.imgur.com/27Ei4MZ.jpg?1" title="Using the console in chrome" />


```python
# Importing our libraries for web parsing
import bs4 as bs
import urllib.request
import pandas as pd
import xlwt
  
# Visiting the website through Python
url = 'https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20170114&end=20180114'
link = urllib.request.urlopen(url).read()
  
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
  dictionary = {'Date' : date, 'Coin' : 'Bitcoin', 'Opening Price'
    :open,'Closing Price' : close, 'Low' :low, 'High': high, 'Volume' : 
    volume, 'Market Cap': marketcap}
  list_of_dic.append(dictionary)
  
# Now we simply put this dictionary into a dataframe
df = pd.DataFrame(list_of_dic)
  
# We drop the first column because we already made a header column
df = df.drop(df.index[0])
  
# We turn our date column into the time of date for easily manipulation
df['Date'] = pd.to_datetime(df['Date'])

print(df.head())
```

      Closing Price     Coin       Date      High       Low       Market Cap  \
    1      13772.00  Bitcoin 2018-01-14  14511.80  13268.00  241,447,000,000   
    2      14360.20  Bitcoin 2018-01-13  14659.50  13952.40  234,391,000,000   
    3      13980.60  Bitcoin 2018-01-12  14229.90  13158.10  225,986,000,000   
    4      13405.80  Bitcoin 2018-01-11  15018.80  13105.90  251,387,000,000   
    5      14973.30  Bitcoin 2018-01-10  14973.30  13691.20  244,981,000,000   
    
      Opening Price          Volume  
    1      14370.80  11,084,100,000  
    2      13952.40  12,763,600,000  
    3      13453.90  12,065,700,000  
    4      14968.20  16,534,100,000  
    5      14588.50  18,500,800,000  


### See how easy that was?
  
   You may be asking yourself, how do we do more than one coin at a time, or how do we change the dates without looking up the url everytime? Well, it is actually pretty simple to do through python functions. 
  
# Part 2: Functions for Ease of Use
***
  
   By using functions, we can easily bind crypto data we have collected and attach prices collected from coinmarketcap in an way that is simple to reuse. Our first function will be wrapping our previously written code into a larger function. We will add the parameters coin for type of coin, start_date for the start of when we want to get prices, and end_date for the final date we want to get prices. We will also set some handy defaults so that if nothing is specified, we will get Bitcoin information during 2017. Notice that we can simply use these parameters to replace the website address, making it very simple to change coin type and date range. We can even add parts to our function so the user can type in there date in almost every format they want to!


```python
def retrieve(coin='Bitcoin', start_date='Jan 1, 2017', end_date='Jan 1, 2018'):
    """Function dedicated to scraping coinmarketcap and retrieving a Pandas DataFrame. The coin and date is able to be
    changed easily"""

    # Ensures proper formatting of user inputs
    coin = coin.lower()
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    start_date = start_date.strftime('%Y%m%d')
    end_date = end_date.strftime('%Y%m%d')


    # Retrieves data with BeautifulSoup and parses
    url = 'https://coinmarketcap.com/currencies/' + coin + '/historical-data/?start=' + start_date + '&end='+end_date
    print(url)
    link = urllib.request.urlopen(url).read()
    soup = bs.BeautifulSoup(link, "html.parser")
    prices_table = soup.find_all('tr')

    # Creates list of dictionaries to feed into the DataFrame
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

    # Creates and sends our DataFrame
    df = pd.DataFrame(list_of_dic)
    df = df.drop(df.index[0])
    df['Date'] = pd.to_datetime(df['Date'])
    # df.set_index(['Date','Coin'], inplace = True)
    return df

#Testing
retrieve('ethereum','December 01, 2017','12/5/2017')

```

    https://coinmarketcap.com/currencies/ethereum/historical-data/?start=20171201&end=20171205





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Closing Price</th>
      <th>Coin</th>
      <th>Date</th>
      <th>High</th>
      <th>Low</th>
      <th>Market Cap</th>
      <th>Opening Price</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>463.28</td>
      <td>Ethereum</td>
      <td>2017-12-05</td>
      <td>473.56</td>
      <td>457.66</td>
      <td>45,212,800,000</td>
      <td>470.29</td>
      <td>1,216,720,000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>470.20</td>
      <td>Ethereum</td>
      <td>2017-12-04</td>
      <td>474.78</td>
      <td>453.31</td>
      <td>44,795,700,000</td>
      <td>466.05</td>
      <td>1,005,550,000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>465.85</td>
      <td>Ethereum</td>
      <td>2017-12-03</td>
      <td>482.81</td>
      <td>451.85</td>
      <td>44,560,500,000</td>
      <td>463.70</td>
      <td>990,557,000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>463.45</td>
      <td>Ethereum</td>
      <td>2017-12-02</td>
      <td>476.24</td>
      <td>456.65</td>
      <td>44,853,300,000</td>
      <td>466.85</td>
      <td>943,650,000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>466.54</td>
      <td>Ethereum</td>
      <td>2017-12-01</td>
      <td>472.61</td>
      <td>428.31</td>
      <td>42,765,000,000</td>
      <td>445.21</td>
      <td>1,247,880,000</td>
    </tr>
  </tbody>
</table>
</div>



## But wait, there's more

   This function provides an incredibly useful tool, but what if we want to get information for more than one coin at a time? We might want to pass in a list of all the coins we want to get information about. We can make a new function, with the function we just made put inside. This way, we can use our old function as many times as we need to. We have the same parameters as our old function, however we pass in a list of coin names rather than just a single coin name.



```python
def retrievealldata(coinlist, s_date = 'Jan 1, 2017', e_date = 'Jan 1, 2018'):
    """This function uses the retrieve function multiple times
    in order to create a DataFrame with multiple cryptocoins"""

    if coinlist is list:
        raise Exception('Must pass a list')

    alldata = pd.DataFrame()
    for coin in coinlist:
        df = retrieve(coin, s_date, e_date)
        alldata = alldata.append(df)

    alldata = alldata.sort_values(by=['Date'], ascending=True)
    # alldata.Date = alldata.Date.dt.strftime('%b %d, %Y')
    alldata.set_index(['Date', 'Coin'], inplace=True)

    return alldata

retrievealldata(['bitcoin','ETHEREUM'], '5/12/2016', 'May 14, 2016')
```

    https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20160512&end=20160514
    https://coinmarketcap.com/currencies/ethereum/historical-data/?start=20160512&end=20160514





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Closing Price</th>
      <th>High</th>
      <th>Low</th>
      <th>Market Cap</th>
      <th>Opening Price</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th>Date</th>
      <th>Coin</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">2016-05-12</th>
      <th>Bitcoin</th>
      <td>454.77</td>
      <td>454.95</td>
      <td>449.25</td>
      <td>7,028,330,000</td>
      <td>452.45</td>
      <td>59,849,300</td>
    </tr>
    <tr>
      <th>Ethereum</th>
      <td>10.06</td>
      <td>10.51</td>
      <td>9.86</td>
      <td>798,952,000</td>
      <td>10.00</td>
      <td>24,137,700</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">2016-05-13</th>
      <th>Bitcoin</th>
      <td>455.67</td>
      <td>457.06</td>
      <td>453.45</td>
      <td>7,067,270,000</td>
      <td>454.85</td>
      <td>60,845,000</td>
    </tr>
    <tr>
      <th>Ethereum</th>
      <td>10.51</td>
      <td>11.05</td>
      <td>10.02</td>
      <td>804,647,000</td>
      <td>10.06</td>
      <td>31,276,900</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">2016-05-14</th>
      <th>Bitcoin</th>
      <td>455.67</td>
      <td>456.84</td>
      <td>454.79</td>
      <td>7,084,100,000</td>
      <td>455.82</td>
      <td>37,209,000</td>
    </tr>
    <tr>
      <th>Ethereum</th>
      <td>10.24</td>
      <td>10.62</td>
      <td>9.81</td>
      <td>840,395,000</td>
      <td>10.51</td>
      <td>18,808,000</td>
    </tr>
  </tbody>
</table>
</div>



### *We now have a scalable solution to get all of our crypto data. Happy crypto investing!*
