# Git Commands Summary

- Git log --stat ==> highlights statistics about commits 
- git log --graph --oneline branch1 branch2
- git clone ==> copy entire history of all files from a repo.
- git config ==> change settings in git, --global flag ==> apply to all git projects
- git checkout *commit id* ==> reseting all files to this commit 
- git status ==> shows which files changed since last commit 
- git add *file* ==> addes a file to the staging area which is an intermediate area before repo.
- git reset *file* ==> removes the file from the staging area
- git diff (without any arguemnts) ==> highlights the difference between files in staging area & working directory
- git diff --staged ==> highlights the difference between files in staging area & repo.
- git branch *branch name* ==> creates a new branch 
- git branch -d *branch name* ==> deletes the branch label 
- git checkout branch name* ==> switches to a branch with the same code as master
- git checkout -b *new branch name* === git branch then git checkout
- git show *commit id* ==> highlights the diff between this commit & its parent
- git merge --abort ==> Restore files to their state before i started the merge
- git remote add *name instead of origin e.g "upstream" * *original repo link* ==> to add the original repo as fetch source
- git remote -v ==> checks remote repos. for fetching & pushing 
- git pull upstream *branch name*


# Commit Messages 
commit messages consists of three distinct parts separated by a blank line: 
1. The title
2. An optional body
3. An optional footer 

The layout looks like this:
- type 
- body
- footer

### The Type
The type is contained within the title and can be one of these types:

- feat: a new feature
- fix: a bug fix
- docs: changes to documentation
- style: formatting, missing semi colons, etc; no code change
- refactor: refactoring production code
- test: adding tests, refactoring test; no production code change
- chore: updating build tasks, package manager configs, etc; no production code change

### The Subject
Subjects should be no greater than 50 characters, should begin with a capital letter and do not end with a period.

Use an imperative tone to describe what a commit does, rather than what it did. For example, use change; not changed or changes.

### The Body
Not all commits are complex enough to warrant a body, therefore it is optional and only used when a commit requires a bit of explanation and context. Use the body to explain the what and why of a commit, not the how.

When writing a body, the blank line between the title and the body is required and you should limit the length of each line to no more than 72 characters.

### The Footer
The footer is optional and is used to reference issue tracker IDs.

### Continue Commit Messages:
1-Include only files related to the feature i'm implementing 
2-Concise subject
3-Commit body should include all the details of the changes i made
4-Link commit ==> #issue 
5-Always make the commit verbal 


##### Master branch acts as production quality branch that never breaks 
##### detached Head state ==> means current commit 










