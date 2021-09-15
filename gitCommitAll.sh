#!/bin/bash
# Standard simple shell script to quickly commit all changes

echo "Enter your commit text to this push using the format 'sometext': "
					# This will prompt the user
read  COMMIT 				# to input the commit text

# This will initiallize, add all changes, commit and push
git init
git add --all
git commit --all -m "$COMMIT"
git push Marc_Speciale_STEP
