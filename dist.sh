#!/usr/bin/env bash


# ======================================================================
echo " :: Create change log..."
CHANGELOG="CHANGELOG.txt"
echo -e "Change Log\n==========\n" > ${CHANGELOG}
git log --oneline --decorate --graph >> ${CHANGELOG}


# ======================================================================
echo " :: Create package..."
python setup.py bdist_wheel --universal


# ======================================================================
echo " :: Distribute package..."
PYPIRC_EXT=pypirc
if [ -z "$1" ]; then
    for FILE in *.${PYPIRC_EXT}; do
        CHOICE=${FILE%\.*}"|"$CHOICE
    done
    echo -n "choose target ["${CHOICE%?}"]: "
    read PYPIRC
else
    PYPIRC=$1
fi
PYPIRC_FILE=${PYPIRC}.${PYPIRC_EXT}

for FILE in dist/*; do
    if [ -f ${FILE} ] && [ -f ${PYPIRC_FILE} ]; then
        twine upload ${FILE} --config-file ${PYPIRC_FILE}
    fi
done
