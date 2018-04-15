---
layout: single
classes:
  - wide
---


# Work in Progress
### *Jacob Sieber*
### *April 12, 2018*

*During my own journey searching for the 'perfect visual tool' to create visualizations, I spent a large amount of time getting familiar with different solutions. This article serves to provide an idea of the software landscape from someone who has learned the basics of the tools provided.*


Visualizing data is a critical componet in the data science tool belt. A well formed visualization can be the difference inbetween putting the audience to sleep and dramatically shifting the entire audience's perspective. But what is the tool that should be used? It can be quite daunting to sort through all of the options that are available. Should the tool used really matter? After all, the outcome is always a graphic anyways. Is presenting with a default Excel bar chart really such a crime?

As with nearly everything data science, there is always a trade-off that must be made. The primary one here is the amount of time spent creating a graphic and the amount of detail that can be incorporated into a graphic. If you would simply like to see the distribution of the data you are dealing with, all that is necessary is a simple Excel histogram. However, incorporating more detail into the graphic can add a whole new perspective on data. Take for example Hans Rosling's gapminder graphic, the poster child of the utility of visualizations. Insights from this graphic are incredibly intuitive and are relatively exciting comapred to a throwaway scatter plot.

<figure>
<a href="https://www.gapminder.org/world/#$majorMode=chart$is;shi=t;ly=2003;lb=f;il=t;fs=11;al=30;stl=t;st=t;nsl=t;se=t$wst;tts=C$ts;sp=5.59290322580644;ti=2013$zpv;v=0$inc_x;mmid=XCOORDS;iid=phAwcNAVuyj1jiMAkmq1iMg;by=ind$inc_y;mmid=YCOORDS;iid=phAwcNAVuyj2tPLxKvvnNPA;by=ind$inc_s;uniValue=8.21;iid=phAwcNAVuyj0XOoBL_n5tAQ;by=ind$inc_c;uniValue=255;gid=CATID0;by=grp$map_x;scale=log;dataMin=194;dataMax=96846$map_y;scale=lin;dataMin=23;dataMax=86$map_s;sma=49;smi=2.65$cd;bd=0$inds=;modified=60"><img src="https://i.imgur.com/298WG9y.gif" title="GapMinder Chart"/></a>
<figcaption><i>The size of the bubbles indicate the size of a country, the y axis is life expentancy, and the x axis is gdp per captia.
from https://www.gapminder.org/tools/</i></figcaption>
</figure>

*The size of the bubbles indicate the size of a country, the y axis is life expentancy, and the x axis is gdp per captia.*
*from https://www.gapminder.org/tools/*

Hopefully, now the value of detailed visualisations is clear. But, what tools are the best at creating powerful graphics? For visualisations that are both interactive and readily presentable, there are three primary tools that I consider to be the strongest canidates with their own distinct strengths and weaknesses. They are Tableau, Shiny, and JavaScript's D3 library.

<a href="https://www.tableau.com/"><img src="https://i.imgur.com/GYVuGFM.jpg" title="Tableau's Logo" height="300" width = "500" /></a>

# Tableau

Tableau is a tool that is both intinutive and powerful tool to use.  Connecting to a remote server, creating impressive visualizations, and custominzing them all within an interactive dashboard has never been as quick or easy to do. The primary strength here is the simplicity in designing presenetable visualizations. Typically, there is no coding at all involved in projects, aside from a few calculated columns. This means that nearly anyone can pick up Tableau and create

## __Pros__

### GUI Interface

This is the primary strength of tools like Tableau. Almost everything is done through a GUI, meaning that nearly anyone can create insightful visualizations and dashboards in a very short amount of time. If your expertise is in a non-technical field such as accounting or marketing, Tableau can easily be a one-stop-shop for all of your visual needs.

### Powerful Interactivity

Nearly of all Tableau's visuals come built in with interactivitive elements. If your boss wants you to drill down from state to city, only a few clicks are needed. Filtering based on a sub-cateogry or an element can also be done on the fly with zero effort.

### Complex Visual Dashboards

Through a visual interface that guides you every step of the way, attractive dashboards that convey a plethora of information can be created. One great expample is  To get some inspiration from some of the best publicy shared creations, check out [Tableau's Community Gallery.](https://public.tableau.com/en-us/s/gallery/). One great example is this dashboard by [Sarah Bartlett](https://sarahlovesdata.co.uk/2018/04/03/vizzing-european-cities-for-iron-viz-europe/amp/).

<a href="https://public.tableau.com/en-us/s/gallery/european-cities-budget?gallery=votd"><img src="https://i.imgur.com/V7mcU25.png" title="Explore European Cities on a Budget" /></a>


## __Cons__

### Closed source

There are two main drawbacks to being closed source. One, most obviously, is that there is a charge for using Tableau (there is a free version named Tableau Public with some limitations). The other drawback is one that I find much more severe. Because Tableau's development is closed source, it is more difficult for non affiliated programers to add their own contributions to improve Tableau. When you use a tool like D3 to build visuals, you not only have access to what the original developers provide, but also the work of [hundreds of other developers](https://github.com/wbkd/awesome-d3) provide. This means that some more unique charts can be more difficult to implement within Tableau.

That being said, Tableau has deep integration with the programing language R. This means that R scripts, packages, and predictive modeling can be use and machine learning models be used within Tableau's GUI interface. However, given that R already has impressive visualization packages and a smooth dashboard framework, R enthusiasts may find designing visualizations in R a simpler and more robust process.

### Lack of Coding Interface

While the GUI interface is the pivotal selling point of Tableau, designing through a purely visual interface can frustrate many coders. While the GUI is fantastic for quickly creating visuals, it doesn't have the expressive power of a solid programming language. So, 'going of the rails' of what Tableau's developers initialy designed can be a fairly frustrating process. However, very impressive visualizations that are wholly unique and creative are still made on Tableau, just a at a lessor rate than programing languages.

## __Honorable Mention__

### Microsoft BI

This is another data visualiztion tool that functions through a GUI as well, and is currently under a lot of development. Microsoft's visual solution borrows a lot of the framework from Excel, so those familiar with the spreadsheet software should have no issue using this tool.

# R-Shiny

<a href="https://shiny.rstudio.com"><img src="https://i.imgur.com/gwvnTS8.png" title="Shiny" /></a>

R-Shiny is where trade off between difficulty and freedom visualization design begins to become apparent. When RStudio designed this package, the central idea was to create interactive web-based dashboards without needing to learn web-base coding HTML, CSS, and JavaScript. In fact, most of the code in Shiny serves as a wrapper around these web-based langauges. Additionaly, many of the R packages used in creating interactive visuals are merely wrappers around popular JavaScript D3 libraries. So, it makes sense that the capibility of Shiny can be greatly extended by using web-based languages. However, learn HTML , CSS, and JavaScript is not neccessary to get a great deal of mileage out of Shiny.

<a href="https://shiny.rstudio.com/gallery/genome-browser.html"><img src="https://i.imgur.com/e4vsPer.png" title="Mapping Genomes" /></a>
