# Google Cloud Setup for Data Science

April 14th, 2019

*Getting started with the ever popular $300 12-month Google Cloud for Data Science applications*

Tired of frying your laptop and waiting days to train that sweet neural network? Want to freely use some of the latest and greatest hardware for data munging and modeling? Is your mother upset at the large power bill that your machine racks up training models? Well this post is for you. Google Cloud is offering a $300 12-month trial (a valid credit card is required) on all of its cloud services. This means that you can set up virtual machines with multiple CPUs, high memory, and powerful graphics cards while coding in a remote location for free. Because Google Cloud is vast in its functionality, it can be a little difficult to find direct, concise, and updated information about how to get Jupyter Notebooks and Python scripts set up. This post goes through how to get a setup on a customized Google Cloud machine that can service your data science needs. This is a windows tutorial, however it is quite easy to carry over the OS specific information across operating systems. We will create a Google Cloud instance with Anaconda, a graphics card, and Jupyter Notebooks that is connected to VSCode and Github.

*Remember two important rules when making projects in the cloud: backup your project code and turn off the virtual machine once you are done. *

## Applications to Install on your Machine

### Google SDK Tool

[Link]([https://cloud.google.com/sdk/](https://cloud.google.com/sdk/). The Google SDK Tool is the command line tool for Google Cloud. We will use this tool to create our virtual machine instance with all of the bells and whistles that we will need.

### Git

[Link]([https://git-scm.com/download](https://git-scm.com/download). Git is another command line tool that we will use briefly for generating our SSH keys and backing up the projects that are made on the cloud. Git is preinstalled in the virtual machine that we will create.

### VSCode with the Remote Development Extension

[Link](https://code.visualstudio.com/). VSCode with the Remote Development Extension is an easy to set up tool that offers an incredible amount of functionality. This functionality is namely: multiple integrated terminals, click and drag to move files from client to host / vice versa, and the most popular IDE (according to Stack Overflow survey 2019). It is advertised as being a local development experience when working on the cloud. One of the other great features of coding with VSCode is that during periods of cloud instability - you won't lose progress and can still code (without code output). After downloading VSCode, you want to install the Remote Development Extension as seen below.

<img src="https://imgur.com/4doiLst" title="VS Code Setup"/>

* If you are using Windows, you also want to bind SSH to the command line by opening up PowerShell as administrator and typing in ``Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0``. This allows VSCode to use SSH commands. *

### GitHub Account

[Link](https://github.com/). GitHub serves as a great way to store code on the cloud. The only downside is if you do not have a premium account or an .edu email address, you must pay a monthly fee to keep code private. There are also other sites, such as Bitbucket, that can serve the same purpose.

# Step 1: Creating the Instance

If you haven't already, you want to create an account on [https://cloud.google.com/](https://cloud.google.com/) . By using the trial on Google Cloud, the functionality on Google Cloud will be free to use until you reach your $300 limit or have your account for over a year. First things first, projects are a way to keep settings separate across multiple cloud projects. The new project button is in the top left, as seen in the image below.

<img src="https://imgur.com/VCOdUnn" title="New Project"/>

Projects in Google Cloud are typically used to manage permissions and resource allocation. For example, a data scientist may have a project with admin permissions for several co-workers and have another project where she is the only person with admin rights. If you are only working by yourself, one project will be adequate.

The three stacked bars in the top left will allow you navigate through the different pages of Google Cloud. You can also use the search bar at the top of Google Cloud's pages in order to navigate the website.

A graphics card will be necessary to train models like neural networks in a reasonable amount of time. In order to get access to graphics card on Google Cloud: Go to the quotas section type in `GPUs all regions` in the metrics box and be sure to check it in the services. Mouse up to `Edit Quotas`  and increase the quota limit to 1. Submit the request and Google will increase your quota after approval.

<img src="https://imgur.com/RykPWUj" title="Quotas"/>

Now it's time to create the instance through the Google Cloud SDK installed on your local machine. If you chose to put Google Cloud SDK on your path during installation, you can do this from the command line. To get your default configuration type in:

``
$ gcloud init
``

You will be asked to log in and set your defaults.

After you get the SDK tool set-up, you want to create your own virtual machine in order to host your super-charged data science platform. You can do this with a single command with customized arguments to suit your specific needs. The example I use here is based off the [fast.ai instructions]([https://course.fast.ai/start_gcp.html](https://course.fast.ai/start_gcp.html)) and shown below:

``
$ gcloud compute instances create deep-learning --zone=us-east1-b --image-family=pytorch-latest-gpu --image-project=deeplearning-platform-release --maintenance-policy=TERMINATE --accelerator="type=nvidia-tesla-p100,count=1" --boot-disk-size=200GB --metadata="install-nvidia-driver=True" --tags http-server,https-server --preemptible
``

The `--preemptible` argument at the end of the command means that our instance can be shut down by Google if there is a resource strain and can only last 24 hours, but this also means that our virtual machine is much less expensive to run. By using a preemptible instance, a reasonable amount of storage, and *shutting down the instance once you are done*, the $300 dollars in free credit will be more than enough for months’ worth of projects. Storage tends to cost the most in the long run, so be sure not to get too excessive on the storage space. On this type of instance, you can always decide to increase the size of the boot disk (i.e. default disk), but it is much more difficult to decrease the amount of storage space used by an instance.

You can customize your instance further by setting options such as `--custom-cpu 48 --custom-memory 80` to configure the virtual machine exactly as you like. It may seem like a good idea to increase the number of CPUs to a very high number to increase the speed of your applications. However, almost most always in machine learning, the biggest bottleneck is the GPU. There are two reasons this is the case. First, using too many CPUs (logical processors) in parallel may actually slow down training models because of the overhead of constantly sending instructions to each CPU. Secondly, many of the most widely used and most accurate models (neural networks and boosted random forests), are much more efficient running on GPUs.

### Configuring the Virtual Machine's Network

Next we need to a few additional configurations to our instance and our project. First, let’s make sure our project allows our applications to connect through web browser to access the newly created virtual machine. This is all done in the VPC network tab on the google cloud website.

First go to the External IP addresses and set a static IP address, be sure to attach this IP address to the virtual machine that you made by clicking the `In use by` column of the reserved address. We do this so the address of the instance does not change every time we reboot the instance.

<img src="https://imgur.com/5xK79F4" title="Static Address"/>

Next go to firewall rules and add a new rule. Name the new rule, and then make sure you set `Targets` to be 'All instances in the network', `Source IP ranges` to be 0.0.0.0/0 and `Protocols and ports` to be specified protocols and ports with tcp:8888. This allows us to get past the firewall of the virtual machine.

<img src="https://imgur.com/n9UpgEV" title="Firewall Rule"/>

### Checking the Virtual Machine

The Virtual Machine instance page is the page that you will most likely visit the most. This is where you can turn off / on the virtual machine and change some of the settings of the virtual machine.

<img src="https://imgur.com/RU4IlD3" title="VM instances"/>

Go to the Compute Engine tab. Under this tab go to VM instances. You should see the VM instance that you created. If you do not see it, make sure you are in the right project. Click on the name of your project to get the configuration options for the instance. Here you can configure almost everything on your instance. Go down to Firewalls and check that `Allow HTTP traffic` and `Allow HTTPS traffic` are both checked.  This allows us to visit our instance through a web browser.

Now go back to the VM instances page and click on the SSH button to the right of your instance to connect to the server. You may need to install the Google Cloud browser add-on; Google will prompt you. Congratulations, we are now inside our Virtual Machine! Anaconda and NVidia drivers are already installed on the server due our set-up command we used in the Google SDK Shell.

# Step 2: Jupyter Notebook Configuration

We can now configure Jupyter Notebooks to use the virtual machine's resources while using a local web browser.  We are in a Linux server, so we will use Linux commands. If you haven't used Linux commands much before, a great cheat sheet is found [here](https://files.fosswire.com/2007/08/fwunixref.pdf).

First, let's configure Jupyter for remote access. Type in the following command:

``` $ jupyter notebook --generate-config  # generates config file```

It will return the location of the configuration file. Copy the location to the clipboard and type the command below while pasting the location given:

``` $ nano {COPIED_CONFIG_PATH} # opens a text editor on the config file```

This will open up the in terminal text editor. Now type in the code as specified in the image below.

<img src="https://i.imgur.com/jbqSQva.png" title="Jupyter Config"/>

When you are done, press CTRL-X and then ENTER, this will take you back to the terminal. Be sure not to rename the file when you save. Our edits to the configuration file set a new default IP and tells Jupyter not to automatically open a browser on startup.

Let's also set a password on Jupyter for convenience. Use the terminal command: `jupyter notebook password` and follow the instructions. This will save us time when have to open Jupyter Notebooks on the browser.

Now, let's give Jupyter Notebooks a quick test. Type in `Jupyter Lab` (or `Jupyter Notebook` if you prefer the pure notebook version) and you will see Jupyter start up. Now, copy the External IP you see back on the Google Cloud VM instances page and navigate your browser to `{COPIED_EXTERNAL_IP} + :8888/`.  After you type in the password, you will now have a Jupyter Notebook in cloud.

#### Creating an Anaconda Virtual Environment for Jupyter

Having multiple environments helps prevent conflicts and bloated python executables. An example on how to use Anaconda to create a virtual environment for deep learning in Jupyter Notebooks is below.

``

$ conda create -n deep_learning

$ conda activate deep_learning

$ conda install -c pytorch pytorch-cpu torchvision --y

$ conda install -c fastai --y

$ conda install jupyter --y

$ python -m ipykernel install --user --name deep_learning --display-name "Deep_Learning"

``

#### Suggested Project Folder Setup

If you want more than a single project, it is a good idea to set up a project folder. An example of making a project folder and creating a new project is below.

```

$ mkdir projects  #mkdir makes directories

$ cd projects  #cd navigates inside of a directory

$ mkdir TwiceML

$ cd TwiceML

$ mkdir jupyter_notebooks data

$ cd ..

``

# Step 3: SSH Keys, GitHub Backups, and Remote VSCode

Setting up SSH keys is useful for automatically verifying a client computer for a virtual machine host and verifying a virtual machine to GitHub. GitHub is useful for backing up the code on your virtual machine and making changes to code from other machines, local or remote.  The tools in VSCode's Remote Development Extension allows for a “local-like development experience”, with tools such as linting, debugging, and extensions, while working on remote code.

### Creating keys on the Local Machine

We will two public/ private key pairs. The first pair will verify our local machine to the virtual machine. The second pair will verify our virtual machine to GitHub.

First, on your local computer open Git Bash and enter the command ``ssh-keygen``. You will be prompted for the desired path of your newly generated SSH keys. The path I used is ``c:/Users/jack/.ssh/jack_laptop``.  At the location of you specified for your SSH keys, you should find two files. For me, the files are named ``jack_laptop`` and ``jack_laptop.pub``. These two files are the private key and the public key, respectively.

``
$ ssh-keygen
$ Generating public/private rsa key pair.
$ Enter file in which to save the key (/c/Users/jack/.ssh/id_rsa): c:/Users/jack/.ssh/jack_laptop
``

You want Git on your computer to associate this private/ public key paring whenever you try to work with a remote repository. To do this first ensure that your SSH-agent is running. You can do this by typing in:

``
$ eval $(ssh-agent -s)
``

Next, add the SSH private key to the SSH-agent:

``
$ ssh-add c:/Users/jack/.ssh/jack_laptop
``

Now open up the ``.pub`` file and copy the text it to the clipboard. Next navigate back to the Google Cloud Platform website. Under the Compute Engine section, there is a section called Metadata. Navigate to that page. Click the `Edit` button and then paste your key in the key data spot.  At the end you should have something similar to the image below:

https://imgur.com/6K2kvOY

You also want to take that same public key and associate it with your GitHub account. To do this go to settings and then go to the SSH and GPG keys section. Press the New SSH key button, name your key, and then paste your public key again.

### Specifying the SSH Keys on VSCode and GitHub

Now the public/private key that you made has access to your virtual machine. Open visual studio code, press `F1` and type in remote-ssh and then go to `Remote-SSH: Open Configuration File`.  Here you see your settings. Now specify the settings like so:

``
Host Deeplearning  # Name
HostName 34.74.32.141  # Virtual Machine IP
User jack_laptop # Username made in Compute Engine - Metadata
IdentityFile C:\Users\jack\.ssh\jack_laptop.ppk # Location of the private key
``

Now by using the explorer bar, clicking the SSH icon,  and right clicking the name you specified in the configuration file, you can connect to the virtual machine just as you would your own computer. You can even use all of your extensions. The most popular extension for data science is the Python extension, which is able to run python scripts as if they were jupyter notebooks. Due to the way the SSH tunneling works, you can even transfer files via click and drag in between the computer and the instance by using the explorer window.

### Specifying the SSH Keys on VSCode

Now, we want to make SSH keys for our virtual machine and connect both of the virtual machine's and the local machine's keys to GitHub. Having the local machine's keys on GitHub will allow us to pull and push code in between the virtual machine, code saved on GitHub servers, and your local machine.

While still in VSCode, press `F1` and type in `Open New Terminal` and press enter. A terminal will pop up on the bottom of the screen. This terminal has the same sort of access as the SSH client we used from the Google Cloud Console. Just as before, type in ``ssh-keygen`` and specify the desired path for your keys. This time, I specified the path ~/.ssh/vm_deep_learning `. Also, be sure to also add the ssh key to git

``
$ ssh-keygen
$ Generating public/private rsa key pair.
$ Enter file in which to save the key (/home/jack/.ssh/id_rsa): ~/.ssh/vm_deep_learning
`
``
$ ssh-add ~/.ssh/vm_deep_learning
``

Now use the command:

``
$ cat ~/.ssh/vm_deep_learning.pub # Second argument is the path of the public key
``

This will show the public key in the terminal. Copy this key to the clipboard. In the same way as before, we can also add this key to our GitHub account.

### Finished

We are finally done configuring our virtual machine to work well with our local machine! In order to use Git to back up our code, it is helpful to get familiar with the common Git commands. There is a cheat sheet [here](https://www.git-tower.com/blog/git-cheat-sheet) that has served me well in the past. Don't forget to put your data/ folder in .gitignore! The most used commands are:

``

$ git init # Starts a new repository
$ git clone # Clones an exsiting repository
$ git add . # Adds in all of changes made for files
$ git commit # Commits changes added
$ git push # Pushes changes to github
$ git pull # Pulls in changes from github

``

Goodluck on all of your data science projects!