Clinical Knowledge Graph
============================

A Python project that allows you to analyse proteomics and clinical data, and integrate and mine knowledge from multiple biomedical databases widely used nowadays.

* Documentation: `https://CKG.readthedocs.io <https://CKG.readthedocs.io>`_

* GitHub: `https://github.com/MannLabs/CKG <https://github.com/MannLabs/CKG>`_
* Free and open source software: `MIT license <https://github.com/MannLabs/CKG/LICENSE.rst>`_

.. image:: /_static/images/banner.jpg
  


Abstract
------------

Several omics data types are already used as diagnostic markers, beacons of possible treatment or prognosis. Advances in technology have paved the way for omics to move into the clinic by generating increasingly larger amounts of high-quality quantitative and qualitative data.  Additionally, knowledge around these data has been collected in diverse public resources, which has facilitated the understanding of these data to some extent. However, there are several challenges that hinder the translation of high-throughput omics data into identifiable, interpretable and actionable clinical markers. One of the main challenges is the interpretation of the multiple hits identified in these experiments. Furthermore, a single omics dimension is often not sufficient to capture the full complexity of disease, which would be aided by integration of several of them. To overcome these challenges, we propose a system that integrates multi-omics data and information spread across a myriad of biomedical databases into a Clinical Knowledge Graph (CKG).  This graph focuses on the data points or entities not as silos but as related components of a graph. To illustrate, in our system an identified protein in a proteomics experiment encompasses also all its related components (other proteins, diseases, drugs, etc.) and their relationships. Thus, our CKG facilitates the interpretation of data and the inference of meaning by providing relevant biological context. Further, ~. Here we describe the current state of the system and depict its use by applying it to use cases such as treatment decisions using cancer genomics and proteomics.


Cloning and installing
-----------------------

The setting up of the CKG includes several steps and might take a few hours (if you are building the database from scratch). However, we have prepared documentation and manuals that will guide through every step.
To get a copy of the GitHub repository on your local machine, please open a terminal windown and run:

.. code-block:: bash

	$ git clone https://github.com/MannLabs/CKG.git

This will create a new folder named "CKG" on your current location. To access the documentation, use the ReadTheDocs link above, or open the html version stored in the *CKG* folder `CKG/docs/build/html/index.html`. After this, follow the instructions in "First Steps" and "Getting Started".


Features
---------------

* Cross-platform: Mac, and Linux are officially supported.

* Docker container runs all neccessary steps to setup the CKG. 


Disclaimer 
---------------

This resource is intended for research purposes and must not substitute a doctor’s medical judgement or healthcare professional advice.

