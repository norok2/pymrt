#!/usr/bin/env bash

repl() { if [ -n $2 ]; then printf "$1"'%.s' $(seq 1 $2); fi; }
title() { echo -e "\n\n$1"; repl "=" ${#1}; echo -e "\n"; }
subtitle() { echo -e "\n$1"; repl "~" ${#1}; echo -e ""; }

# ======================================================================
title "Update project"

subtitle "Tag History"
git tag
subtitle "Project Status"
git status

NEW_VERSION=`git describe --abbrev=0 --tags`
echo -e -n "\n>> choose new version number [${NEW_VERSION}]: "
read INPUT
NEW_VERSION=${INPUT:-$NEW_VERSION}

MESSAGE="Distribute to PyPI."
echo -e -n "\n>> set commit and tag message [${MESSAGE}]: "
read INPUT
MESSAGE=${INPUT:-"$MESSAGE"}

subtitle "Update project"
git commit -uno -a -m "$MESSAGE"
git tag -f "$NEW_VERSION" -m "$MESSAGE"
git push


# ======================================================================
title "Create change log"
CHANGELOG=CHANGELOG.txt
echo -e "Change Log\n==========\n" > ${CHANGELOG}
git log --oneline --decorate --graph >> ${CHANGELOG}
echo -e "${CHANGELOG} successfully created."


# ======================================================================
title "Create package"
python setup.py bdist_wheel --universal


# ======================================================================
title "Distribute package"
PYPIRC_EXT=pypirc
PYPIRC_FILES=(*.$PYPIRC_EXT)
NUM_PYPIRC_FILES=${#PYPIRC_FILES[@]}
if [ -z "$1" ]; then
    if [ "$NUM_PYPIRC_FILES" -gt 1 ]; then
        for FILE in ${PYPIRC_FILES[@]}; do
            CHOICE=${FILE%\.*}
            CHOICES=${CHOICE}"|"${CHOICES}
        done
        echo -e -n "\n>> available targets: ["${CHOICES%?}"]"
        echo -e -n "\n>> choose target ["${CHOICE}"]: "
        read INPUT
        PYPIRC=${INPUT:-$CHOICE}
    else
        PYPIRC_FILE="${PYPIRC_FILES[0]}"
    fi
else
    PYPIRC=$1
fi
if [ -z ${PYPIRC_FILE} ]; then
    PYPIRC_FILE=${PYPIRC}.${PYPIRC_EXT}
fi
echo -e "(use config file: $PYPIRC_FILE)"

function twine_upload() {
    if [ -f $1 ] && [ -f ${PYPIRC_FILE} ]; then
        subtitle "Uploading \`$1\`"
        twine upload --config-file "$PYPIRC_FILE" "$1"
    else
        subtitle "Skipping \`$1\`"
    fi
}

DISTS_FILES=(dist/*)
NUM_DISTS_FILES=${#DISTS_FILES[@]}
if [ "$NUM_DISTS_FILES" -gt 1 ]; then
    subtitle "Available dist files"
    ( IFS=$'\n'; echo -e "${DISTS_FILES[*]}" )
    echo -e -n "\n>> Process only last file [YES/no] (otherwise, all files): "
    read INPUT
    ONLY_LAST=${INPUT:-yes}
    if [ "$ONLY_LAST" = "no" ]; then
        for FILE in ${DISTS_FILES[@]}; do
            twine_upload "$FILE"
        done
    fi
fi

if [ "$NUM_DISTS_FILES" -eq 1 ] || [ -z $ONLY_LAST ] || [ $ONLY_LAST = "yes" ]; then
    twine_upload "${DISTS_FILES[${#DIST_FILES[@]} - 1]}"
fi
