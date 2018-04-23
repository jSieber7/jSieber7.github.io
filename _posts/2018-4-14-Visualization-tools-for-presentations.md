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

# Tableau
<a href="https://www.tableau.com/"><img src="https://i.imgur.com/GYVuGFM.jpg" title="Tableau's Logo" height="300" width = "500" /></a>

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

R-Shiny is where trade off between difficulty and freedom in visualization design begins to become apparent. When RStudio designed this package, the central idea was to create interactive web-based dashboards for analysis all through R. Instead of creating new web functionailty in R, most of the code in Shiny serves as a wrapper around the web-based langauges html, css, and JavaScript. Additionaly, many of the R packages used in creating interactive visuals are merely wrappers around popular JavaScript D3 libraries. So, it makes sense that the capibility of Shiny can be greatly extended by using web-based languages. However, knowledge of HTML, CSS, and JavaScript are not neccessary to get a great deal of mileage out of Shiny.



## __Pros__

### Strong Data Analysis Stack

With all of the powerful and unique packages for R, data transformation and modeling are easily integrated into Shiny dashboards. Almost any machine learning model has methods to visualize metrics. Rather than exporting visualizations to another visual tool, R can filter, transform, model and visualize data on the fly within the dashboard.

### Abundance of Visualization R Packages

There are hundreds of packages in R that have a visualization component to them. If you are using a tool like Tableau or Power BI, if the developers did not explicitly program the visual tool themselves, you will have to use another tool. R packages contain the work of hundreds of different contributers and can feature some of the lastest in research. Many advanced visualizations have a leaf 

<a href="https://shiny.rstudio.com/gallery/genome-browser.html"><img src="https://i.imgur.com/e4vsPer.png" title="Mapping Genomes" /></a>


## __Cons__

### Learning Curve

While R is a relatively simple language to operate in, it is still it's own programming language. Learning R and the framework for Shiny can take some time, even more if you wish to learn to build an visual geom mapper from scratch.

### Limited comapred to JavaScript

R-Shiny provides a relatively simple way to create web-apps to share visuals and analysis, however many of the greatest visual tools are contained within JavaScript. Many of the most advanced R web-visualizations are wrappers around JavaScript. There are many great visuals that JavaScript has in D3 that R does not have a package for. 


# JavaScript

<a href="https://www.javascript.com/"><img src="https://i.imgur.com/hR3vzAQ.png" title="JavaScript" /></a>

The reigning king of web visualization. Through JavaScript and D3, one of a kind visualizations can be created in an open source environment. JavaScript has an incredible amount of libraries that can be utilized in order to get the right visual 

Whenever you see an interactive visual 

## __Pros__

### The language based on the Interactive Web

<a href="http://christophermanning.org/projects/voronoi-diagram-with-force-directed-nodes-and-delaunay-links"><img src="https://i.imgur.com/tkRon6a.png" title="Voronoi Diagram with Force Directed Nodes and Delaunay Links" /></a>

Through D3 and other popular JavaScript libraries, professional web app visualizations can be made. JavaScript allows anyone with a current web browser to view visualizations custom built from the ground up. These custom visualizations can be built with much more cust

### A greater amount of visual tools to use

Compared to other tools such as R, Python, or Tableau, JavaScript has more comprenhensive libraries decidated to creating interactive web-visualizations. This allows an experienced JavaScript coder to create a wider range of visualizations with more intutive interactivity. 


## __Cons__


### Lack of true data anylsis tools

While JavaScript does a few libraries decicated to munging through data, true data anaylsis must be done through other means, such as R or Microsoft Azure. There is no nice visual integrations with model building here. Decicating the time to learning a new software language purely for web-interactive visualizations may not be the best use of time for someone who is busy, espcially since many of the several popular JavaScript visualizations retain most of their functionailty through R.

### Learning Curve

JavaScript isn't a terriblly difficult language to learn, however it will take time to become accusomed to creating visualizations. Custom built visuals in D3 can often times contain a great deal of code, and require a deep understanding of the D3 framework. While JavaScript does have some utility useful outside of visualizations, it is mostly in web-development. So, analysts may not be able to get as much milage out of JavaScript as they could with R, Python, or Java.

# Conclusion

As seen throughout the post, different data visualization tools have different strengths and weakness. Moreover, these different tools can be integrated with each other. This allows the different strengths of these tools to complement one another and weakness to be compensated for. Ideally, someone well versed in visualization would have a brief working knowledge with these tools.

