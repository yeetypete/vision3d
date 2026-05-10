# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?= -W --keep-going -n -j auto
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Live local rendering: rebuilds on save and opens the page in your browser.
# --watch picks up edits outside SOURCEDIR (gallery scripts, vision3d docstrings).
# --ignore breaks rebuild loops on paths Sphinx writes back into the source tree
# (sphinx-gallery's MD5/output cache, autosummary's generated stubs).
livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)" \
		--open-browser \
		--watch ../gallery \
		--watch ../src \
		--ignore "$(SOURCEDIR)/auto_examples/*" \
		--ignore "$(SOURCEDIR)/api/generated/*" \
		$(SPHINXOPTS) $(O)

.PHONY: help livehtml Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
