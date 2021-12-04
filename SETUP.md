# Setup Instructions
Pre-requisites: 
* Python 3.9
* Git

_Note: This instruction document assumes a Windows OS. Commands may vary on your OS. Google your equivalent commands if you are using another OS_

Type the commands
```
python --version
git --version
```
To confirm you meet the requirements. If you do not, see https://python.org/ and https://git-scm.com/ for instructions for installation/upgrading. 

### Cloning the repo
Open your computer's command prompt. 
Navigate to the directory you wish to clone the repo using the command, and clone the repo using the following lines
```
cd /path/to/directory
git clone https://github.com/UTRA-ART/CV-Pipeline.git 
```
You should see the repository in your file system now. 


### Setting up venv 
The main branch of this repo contains a requirements.txt file for easily installing dependencies. Run the following commands to set up your virtual environment
```
cd /path/to/repository 
python -m venv venv
```
Activate the venv by running the 
At this point, your command terminal should show
```(venv) /path/to/repository>``` where the (venv) indicates that your virtual environment is running. 
If you do not see this, don't continue. To install the required packages, run 
```pip install -r requirements.txt```. 
Confirm the packages were successfully installed using 
```pip list```.


### Creating a development branch 
It's good practice, and required for this subsection, to not work directly on the main branch. This ensures that everyone working on the project has 
stable code to work off of while they develop. To create your own branch, run the following commands
```
cd /path/to/repository 
git checkout -b new_branch_name
```
Confirm you have move to this new branch by running
```git branch```. 
To push this branch to the github the first time, use
```git push --set-upstream origin branch_name```, 
and all future times just use 
```git push```. 
You can see all branches at anytime by using ``` git branch ``` and switch branches using ``` git checkout branch_name ```. 
#### MAKE SURE YOU ARE ON THE CORRECT BRANCH WHEN YOU ARE WORKING

Now your system is set up! If you are new to using git, please work through the team's github tutorial to learn how to use the software, if not, start coding! 