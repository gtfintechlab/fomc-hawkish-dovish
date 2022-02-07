## Contributing

### Setup
If this is your first time contributing to the project, clone repository to your local device:

Clone the repo. 
```
git clone https://github.com/gtfintechlab/fomc-hawkish-dovish
cd fomc-hawkish-dovish
```

Now your local repo will have one remote corresponding to cloned repo in your github account.
```
git remote -v
origin	https://github.com/gtfintechlab/fomc-hawkish-dovish (fetch)
origin	https://github.com/gtfintechlab/fomc-hawkish-dovish (push)
```

### Conda environment

Create a Conda environment from the provided `environment.yml` file:

`conda env create --file environment.yml`

This only needs to be done once.

Now you are good to go!

To activate environment:

`conda activate fomc-hawkish-dovish-env`

If you install any new dependencies, change the environment.yml file manually. Do not use `conda env export` because it will create cross-platform compatibility issues. 

For detailed conda commands check: [Conda Cheat Sheet](https://github.com/gtfintechlab/onboarding-docs/blob/main/conda_cheatsheet.md)

### Working on a new feature or updating the existing codebase

To start development on a new feature, first take a pull of latest changes from origin repository.
```
git checkout main
git pull --rebase origin main
```
Other option to update your local main branch to origin main branch is as follows: 
```
git checkout main
git fetch origin main
git reset --hard origin/main
```
It is not advisable to use second approach, don't use it if you don't know what you are doing. 

**You should reset your local main branch only if you don't have any important changes/features on local main branch which are not on origin main branch. This will reset your local main branch to look exactly same as origin main branch (main branch on core repo).**

Now your local main branch is same as origin main branch. To start working on a new feature, create a new feature branch from local main branch.
```
git checkout -b feature/feature_name
```

Start working on your feature, add changed files and commit them. Let's say you add/update two files `file1` and `file2`. You have to add these files to git staging area using: 
```
git add file1 file2
```

You can also do `git add .` to add all changed files to git staging area. Once this is done, you need to commit these changes.
```
git commit -m "Commit message here"
```

Push these changes to your branch on online/remote repo
```
git push -u origin feature/feature_name
```
On the GitHub web client, create a new pull request (PR). Set the PR type to `Draft pull request`. Append `WIP:` (work in progress) to the beginning of your PR's name. This indicates that your feature is still being developed and is not yet ready for review.

In case you want to update your pull request, you can push the changes to the same branch corresponding to branch in PR. Changes pushed will be reflected in the PR.

If your feature branch is ready for review, in the GitHub web client, go to your PR, remove the `WIP:` prefix from the PR name, and assign it to a teammate for review. When ready to merge, setup a call with project leader to `Rebase and merge`.

### Centralized Workflow

This project uses a **Centralized Workflow** meaning that there is a central repository that serves as the single point-of-entry for all changes to the project.

See [here](https://www.atlassian.com/git/tutorials/comparing-workflows) for more information on the centralized workflow. The Example section is especially useful to visualize how this works.


### Branches

There is one permanent branch: `main`. The `main` branch contains the most recent stable release. No commits should be made directly to the main branch. Commit will be merged in `main` branch through PR only. 

Temporary `feature` branches can exist which contain features that are being actively worked on. Once merged, these feature brances should be deleted. Feature branches have the following naming convention: `feature/<feature_name>`

### Commit Messages

See [here](https://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html) for commit message guidelines. Commit messages should be short, should be written as commands, and should not have ending punctuation.

Example of a good commit message: `Add readme`

Example of an bad commit message: `Adds readme.`

### Updating pull requests

A pull request should not contain unnecessary commits. If you add more commits to PR after someone requests a review, please do not 
add commits like `requested changes` or `changed as requested`. Instead, you should squash such unnecessary commits in original 
commit message. Squashing will combine both commits into one, and you can give a meaningful commit message to final commit message. 

For example, if you want to merge last 5 commits, you will have to run
```shell
git rebase -i HEAD~5
```

This will open your git editor. Change all `pick` words to `squash` words except for the one commit (generally the topmost commit the editor, since you can anyways change the 
commit message). Save and close this file. 

After this, one more editor window will open. Here, you can change the commit message, and provide whatever message you want. The first
non comment line will become your final commit message. 

Please note that you should only squash your commits. Also, 
please do not perform interactive rebase on merge commits.

This way, you can update your commit history in git. Squashing commits is a good practice followed by almost every good open source 
project. To read more about rewriting git history, refer the official documentation [here](https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History)