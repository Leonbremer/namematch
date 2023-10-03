# Readme - Fuzzy name matching

## The algorithm

This project proposes a fuzzy name matching algorithm that uses existing computational methods. The algorithm takes four steps:
1. cleaning of character strings;
1. similarity scoring;
1. a decision rule based on a supervised machine learning method; and
1. disambiguation of the resulting matching links.

### Step 1: Cleaning

For the cleaning several functions are created. Strings are normalized, but also legal names like *incorporated* are recognized and either abbreviated or deleted.

### Step 2: Similarity scoring

Similarity scoring is achieved by vectorizing the firm names using the TF-IDF method and sequentially calculating the angles between each vectorized firm name across databases using the Cosine Similarity. A cutoff angle is chosen to select a limited set of similar firm name matches, referred to as *candidates*.

### Step 3: Decision rule

Not all candidates are correct matches. Therefore a random sample of candidates is manually labelled. A probit regression is used to fit a model explaining true match status with name and link characteristics. A part of the labelled sample is used to create performance indicators. The cutoff on the predicted probability ($\bar{p}$) of a true match is determined by the maximum F1 score. The unlabelled set of candidates then gets their probability of being a true match predicted ($\hat{p}$) and any candidate for which $\hat{p} > \bar{p}$ is considered a match.

### Step 4: Disambiguation

The resulting linkages suffer from multiplicity, meaning that applicants can link to multiple firms. This is for most follow-up applications not desirable. This last step provides three solutions for disambiguation. First, the applicant's link with the highest $\hat{p}$ is chosen, leaving each applicant linked to one single firm. Second, a community detection algorithm determines which firms belong to the same group based on their co-occurences in links with patent applicants. If applicants are linked to multiple groups they are assigned to the group with most firms (this only occurs infrequently). Third, the largest consolidated firm in the firm group (community) can be taken to represent the entire firm group. Applicants are then uniquely linked to one consolidated firm.

## The application

The algorithm is applied to matching the Amadeus Financials database and the Amadeus Subsidiaries database, both containing data from financial statements, to PATSTAT (Autumn 2018 edition), a worlwide patent database. The algorithm vastly outperformes a simple exact name match. The number of matched firms more than doubles. The number of matched applicants increases with more than 400%. And 25.5% of patent applications since 1950 have at least one applicant that can be linked to a firm in the Amadeus data, compared to 3.6%.

## The code

The code is a work in progress. Unfortunately I cannot share the data of the application. The module parameters.py contains personal paths to files and is therefore omitted here.

## Resulting data

The results of the matching exercise are also made available here. The data is split in separate CSV files. Files are zipped.

One file includes the links between patent applicants (firms only) and Amadeus firms (from both the Amadeus Financials and Amadeus Subsidiaries databases), including some matching statistics. The file links person_id (PATSTAT) to IDNR and SUBS_BVDEPNR (Amadeus Financials and Subsidiaries, respectively). The Amadeus identifiers should match the Orbis (Bureau van Dijk) identifiers. Other columns indicate to which Amadeus source the Amadeus ID refers (amadeus_source_var), the internal link_id (self-assigned), an indication whether the link was drawn at random for manual labelling (randomlabelled), info on whether the link is a true match (either labelled or because the names are the same after cleaning) (truematch), whether the link is not an exact match (nonexact), which role the link played in the machine learning process (training, checking, out-of-sample prediction) (mlpurpose), model prediction (prediction), model's decision (decision_model), and final decision (decision) (this database only contains the links that have decision=True).

A second file contains the links between the patent applicant (person_id) and the constructed community ID (community_id). Details are in the paper (see below). In short: the Amadeus IDs are reliable, the PATSTAT IDs are not. The same patent applicant often occurs in the PATSTAT data with multiple identifiers (person_id). Also, the matching exercise does not guarantee 1-on-1 links between patent applicants and firms. To disambiguate these links, the firm-applicant links are used to detect disjoint communities of patenting firms (IDNR and SUBS_BVDEPNR). When using these communities, each Amadeus firm is assigned to only one community (potentially multiple firms link to one community). This makes the data more useful for further analysis, as it avoids double counting of patenting activities by firms. Any follow-up analysis would need to be on this community level.

A third file uses the constructed communities and searches for the most dominant Amadeus firm in the community with consolidated data. This way one could do analysis with only large consolidated firms and their patents.

A fourth file is the result from an alternative disambiguation exercise. It takes all links and uses the predicted probability of links being a true match ($\hat{p}$ in the paper or *prediction* in the data) and for each patent applicant only keeps the link with the highest probability. It only considers Amadeus firm data that is not reported on the consolidated level. It therefore tries to link patent applicants to an unconsolidated firm (and not a large firm group).

## Accompanying article and referencing

This work is also formalized in an academic article. It is currently available as a [Tinbergen Institute Discussion paper](https://tinbergen.nl/discussion-paper/6276/23-055-viii-fuzzy-firm-name-matching-merging-amadeus-firm-data-to-patstat). When using the code or the data, please reference this article.
