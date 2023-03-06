# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = '深度学习在建筑工程中的应用'
copyright = '2023, 重庆大学智能建造实验室'
author = '重庆大学智能建造实验室'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx.ext.todo",
]
todo_include_todos = True
templates_path = ['_templates']
exclude_patterns = []

language = 'zh_CN'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]
myst_url_schemes = ("http", "https", "mailto")

# sphinx-book-theme 配置
html_theme_options = {
    "repository_url": "https://github.com/IBLofCQU/book001",
    "use_repository_button": True,
    "home_page_in_toc": False,
    "show_navbar_depth": 3,
    "toc_title": project,
    "show_toc_level": 3,
    "use_sidenotes": True,
    "use_issues_button": True,
}
html_title = project

html_css_files = [
    'css/expands.css',
]
