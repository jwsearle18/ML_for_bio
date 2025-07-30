library(reticulate)

reticulate::py_install("transformers")
reticulate::py_install("tf-keras")
reticulate::py_install("sentencepiece")

transformers = reticulate::import("transformers")
sentencepiece = reticulate::import("sentencepiece")

qa_pipeline <- transformers$pipeline("question-answering")

abstract1 = "Predicting the blood–brain barrier (BBB) permeability of small-molecule compounds using a novel artificial intelligence platform is necessary for drug discovery. Machine learning and a large language model on artificial intelligence (AI) tools improve the accuracy and shorten the time for new drug development. The primary goal of this research is to develop artificial intelligence (AI) computing models and novel deep learning architectures capable of predicting whether molecules can permeate the human blood–brain barrier (BBB). The in silico (computational) and in vitro (experimental) results were validated by the Natural Products Research Laboratories (NPRL) at China Medical University Hospital (CMUH). The transformer-based MegaMolBART was used as the simplified molecular input line entry system (SMILES) encoder with an XGBoost classifier as an in silico method to check if a molecule could cross through the BBB. We used Morgan or Circular fingerprints to apply the Morgan algorithm to a set of atomic invariants as a baseline encoder also with an XGBoost classifier to compare the results. BBB permeability was assessed in vitro using three-dimensional (3D) human BBB spheroids (human brain microvascular endothelial cells, brain vascular pericytes, and astrocytes). Using multiple BBB databases, the results of the final in silico transformer and XGBoost model achieved an area under the receiver operating characteristic curve of 0.88 on the held-out test dataset. Temozolomide (TMZ) and 21 randomly selected BBB permeable compounds (Pred scores = 1, indicating BBB-permeable) from the NPRL penetrated human BBB spheroid cells. No evidence suggests that ferulic acid or five BBB-impermeable compounds (Pred scores < 1.29423E−05, which designate compounds that pass through the human BBB) can pass through the spheroid cells of the BBB. Our validation of in vitro experiments indicated that the in silico prediction of small-molecule permeation in the BBB model is accurate. Transformer-based models like MegaMolBART, leveraging the SMILES representations of molecules, show great promise for applications in new drug discovery. These models have the potential to accelerate the development of novel targeted treatments for disorders of the central nervous system."

abstract2 = "The widespread adoption of electronic health record (EHRs) in healthcare systems has created a vast and continuously growing resource of clinical data and provides new opportunities for population-based research. In particular, the linking of EHRs to biospecimens and genomic data in biobanks may help address what has become a rate-limiting study for genetic research: the need for large sample sizes. The principal roadblock to capitalizing on these resources is the need to establish the validity of phenotypes extracted from the EHR. For psychiatric genetic research, this represents a particular challenge given that diagnosis is based on patient reports and clinician observations that may not be well-captured in billing codes or narrative records. This review addresses the opportunities and pitfalls in EHR-based phenotyping with a focus on their application to psychiatric genetic research. A growing number of studies have demonstrated that diagnostic algorithms with high positive predictive value can be derived from EHRs, especially when structured data are supplemented by text mining approaches. Such algorithms enable semi-automated phenotyping for large-scale case-control studies. In addition, the scale and scope of EHR databases have been used successfully to identify phenotypic subgroups and derive algorithms for longitudinal risk prediction. EHR-based genomics are particularly well-suited to rapid look-up replication of putative risk genes, studies of pleiotropy (phenomewide association studies or PheWAS), investigations of genetic networks and overlap across the phenome, and pharmacogenomic research. EHR phenotyping has been relatively under-utilized in psychiatric genomic research but may become a key component of efforts to advance precision psychiatry."
 
abstract3 = "The genome is a sequence that encodes the DNA, RNA, and proteins that orchestrate an organism’s function. We present Evo, a long-context genomic foundation model with a frontier architecture trained on millions of prokaryotic and phage genomes, and report scaling laws on DNA to complement observations in language and vision. Evo generalizes across DNA, RNA, and proteins, enabling zero-shot function prediction competitive with domain-specific language models and the generation of functional CRISPR-Cas and transposon systems, representing the first examples of protein-RNA and protein-DNA codesign with a language model. Evo also learns how small mutations affect whole-organism fitness and generates megabase-scale sequences with plausible genomic architecture. These prediction and generation capabilities span molecular to genomic scales of complexity, advancing our understanding and control of biology." 

qa1_1 <- qa_pipeline(list(question = "How good are AI tools at predicting blood-brain barrier permeability?", context = abstract1))
qa1_2 <- qa_pipeline(list(question = "How does the study improve drug candidate selection?", context = abstract1))

qa2_1 <- qa_pipeline(list(question = "What is the importance of this study?", context = abstract2))
qa2_2 <- qa_pipeline(list(question = "What is the main resource used for phenotyping in this article?", context = abstract2))

qa3_1 <- qa_pipeline(list(question = "What is the name of the long-context genomic foundation model that they present?", context = abstract3))
qa3_2 <- qa_pipeline(list(question = "What is the genome?", context = abstract3))

print(qa3_1$answer)
print(qa3_2$answer)

print(qa1_1$answer)
print(qa1_2$answer)

print(qa2_1$answer)
print(qa2_2$answer)

english_abstract <- "Accumulating evidence points to the impact of the gut microbiota in regulating various chronic inflammatory disorders such as cancers. The intestinal microbiome is not only influencing the spontaneous course of colon malignancies but also acts at distant sterile sites of neoplasia, mostly playing a detrimental role. By providing microbial-associated molecular patterns and potentially antigens sharing molecular mimicry with tumor antigens, our commensals modulate the local and the systemic immune tonus, eventually influencing tumor microenvironment. Complicating this algorithm, therapeutic interventions alter the delicate balance between the epithelium, the microbial community, and the intestinal immunity, governing the final clinical outcome. This seminar focused on the impact of the intestinal composition on the immunomodulatory and therapeutic activities of distinct compounds (alkylating agents, platinum salts and immunotherapies) used in oncology. This research opens up “the era of anticancer probiotics” aimed at restoring gut eubiosis for a better clinical outcome in cancer patients."

french_abstract <- "Récemment, l’impact du microbiote intestinal dans diverses pathologies inflammatoires chroniques, dont le cancer, a été mis en évidence. Le microbiome intestinal régule l’évolution spontanée des tumeurs malignes du côlon et aussi la carcinogenèse extra-intestinale, jouant principalement un rôle délétère. En exprimant des motifs moléculaires associés aux microbes et, potentiellement, des antigènes partageant un mimétisme moléculaire avec des antigènes tumoraux, nos commensaux modulent le tonus immunitaire local et systémique, et peuvent influencer le microenvironnement tumoral. Compliquant cette interaction, les traitements contre le cancer altèrent l’équilibre entre épithélium, microbiote et immunité intestinale, dictant ainsi la réponse clinique. Ce séminaire se concentre sur l’impact de la composition du microbiote intestinal sur les propriétés thérapeutiques et immunomodulatrices de différents agents (agents alkylants, sels de platine et immunothérapies) utilisés en oncologie. Ce champ de recherche ouvre les portes vers « l’ère des probiotiques anti-cancer » visant à restaurer une eubiose intestinale, de manière à améliorer la réponse clinique des patients atteints de cancer."

fr_translator <- transformers$pipeline(
  task = "translation",
  model = "Helsinki-NLP/opus-mt-en-fr"
)

en_translator <- transformers$pipeline(
  task = "translation",
  model = "Helsinki-NLP/opus-mt-fr-en"
)

fr_translation <- fr_translator(english_abstract)
fr_translated_text <- fr_translation[[1]]$translation_text

en_translation <- en_translator(fr_translated_text)
en_translated_text <- en_translation[[1]]$translation_text

cat("Original English Abstract:\n", english_abstract, "\n\n")
cat("Translated to French:\n", fr_translated_text, "\n\n")
cat("Translated Back to English:\n", en_translated_text, "\n")
