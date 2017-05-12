# Following post: using-ipython-notebooks-under-version-control from SO
# Minor changes to make changes local instead of system wide

git config --local core.attributesfile .gitattributes
git config --local filter.dropoutput_ipynb.clean ipynb_output_filter.py
git config --local filter.dropoutput_ipynb.smudge cat

