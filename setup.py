from distutils.core import setup
setup(
  name = 'splashlab',         # How you named your package folder (MyLib)
  packages = ['splashlab'],   # Chose the same as "name"
  version = 'v0.0.3',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'A package for fluid mechanic experimentalists',   # Give a short description about your library
  author = 'Spencer Truman',                   # Type in your name
  author_email = 'trumans24@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/FluidsLab/SplashLab',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/FluidsLab/SplashLab/archive/refs/tags/v0.0.3.tar.gz',    # I explain this later on
  keywords = ['Fluid Dynamics', 'Experiment'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'numpy',
          'pandas',
          'sympy',
          'matplotlib',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.8',
  ],
)