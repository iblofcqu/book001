# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = '深度学习再建筑工程中的应用'
copyright = '2023, 重庆大学智能建造实验室'
author = '重庆大学智能建造实验室'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
]

templates_path = ['_templates']
exclude_patterns = []

language = 'zh_CN'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
# myst_enable_extensions = [
#     "amsmath",
#     # "attrs_inline",
#     "colon_fence",
#     "deflist",
#     "dollarmath",
#     "fieldlist",
#     "html_admonition",
#     "html_image",
#     # "inv_link",
#     # "linkify",
#     "replacements",
#     "smartquotes",
#     "strikethrough",
#     "substitution",
#     "tasklist",
# ]

# sphinx-book-theme 配置
html_theme_options = {
    "repository_url": "https://github.com/IBLofCQU/book001",
    "use_repository_button": True,
    "home_page_in_toc": True,
    "show_navbar_depth": 2,
    "toc_title": project,
    "show_toc_level": 2
}
html_title = project

# html_sidebars = [
#     ""
# ]
