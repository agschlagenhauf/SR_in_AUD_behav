# Multi-stage decision-making task

This folder contains JavaScript/HTML/CSS code for the multi-stage decision-making task.

## How to use
All code is written for JATOS (https://www.jatos.org/). You can test the code in a local JATOS installation. To do so, 
- save the respective folder in the study_assets_root directory of your local JATOS installation
- open the loader.bat script
- open local JATOS in a browser
- click study links and create a link
- as code is written to include Prolific IDs and other subject identifiers, you'll need to manually add the respective parts onto your link (e.g. `?participant=30620126hqIHzHP2GhTvxYt`; see `main.js` function `prepareComponentFlow()` for more details on how IDs are extracted)

To host the code on a public server, you'll need to export the study from your local JATOS installation and import it into the server JATOS version. 