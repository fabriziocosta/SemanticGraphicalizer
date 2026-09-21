# Ontology-Guided Text Graphicalization

## A Hierarchical Semantic Compilation Framework for Scientific Documents

### Abstract

This white paper proposes a framework for transforming long-form scientific text into structured, ontology-grounded knowledge graphs. The central idea is to avoid extracting graph nodes and relations directly from unconstrained text. Instead, the system progressively transforms a document through a sequence of increasingly structured representations: document segmentation, information-preserving summarization, ontology-conditioned semantic normalization, proposition decomposition, triple construction, and graph integration.

The proposed architecture treats an ontology as more than a set of labels to be independently detected. The ontology provides a conceptual vocabulary through which relevant information is expressed and eventually represented as a graph. This creates an intermediate semantic layer between natural language and symbolic graph structure.

The framework is designed around three objectives: preservation of relevant information, explicit control over the conceptual vocabulary used to describe that information, and traceability from graph assertions back to the source text. It is intended particularly for scientific documents, where document structure, specialized terminology, methodological descriptions, and relations among concepts create difficulties for direct entity and relation extraction.

The principal research hypothesis is that graph construction can become more reliable and interpretable when it is treated as a process of progressive semantic compression rather than as a collection of independent classification decisions.

---

## 1. Motivation

The objective of text graphicalization is to transform textual information into an explicit graph representation in which entities, concepts, events, properties, and other relevant objects are represented as nodes and their relationships as edges.

A natural initial architecture is to approach this as a classification problem. Given an ontology containing concepts

$$
O_C=\{c_1,c_2,\ldots,c_n\},
$$

a model can independently determine whether each concept is represented in a passage. Once a subset of concepts has been identified, a second classifier can evaluate pairs of detected concepts and determine whether an ontology-defined relation connects them.

Encoder models provide a straightforward implementation of such an architecture because a pretrained representation can be connected to task-specific classification output layers. BERT, for example, was explicitly introduced as a bidirectional Transformer encoder that could be fine-tuned for downstream tasks by adding an output layer.

This produces a pipeline of the general form

$$
\text{Text}
\rightarrow
\text{Concept Classification}
\rightarrow
\text{Pairwise Relation Classification}
\rightarrow
\text{Graph}.
$$

The proposed work starts from a different formulation. Instead of asking whether each ontology concept independently appears in a passage, the system asks:

**How should the information in this text be expressed when viewed through the conceptual framework supplied by the ontology?**

The difference is fundamental. The ontology becomes a language for representing the text rather than simply a catalogue against which the text is classified.

---

## 2. Limitations of Direct Classification

The original classification formulation creates several design problems that motivate the proposed architecture.

First, concept detection and relation detection are separated even when their interpretations are mutually dependent. A concept may become identifiable only in the context of a relation, while the appropriate relation may depend on the semantic type assigned to its arguments.

Second, independently evaluating candidate relations creates a rapidly increasing number of possible comparisons as the number of candidate nodes increases. If \(n\) concepts are present, pairwise evaluation can involve a number of candidate pairs proportional to \(n^2\).

Third, errors at the concept-detection stage constrain everything that follows. A missing concept cannot participate in any later relation, while an incorrectly introduced concept creates additional candidate relations.

Most importantly, the desired output is a structured semantic object, while independent classifiers produce a set of local decisions. The proposed system therefore introduces intermediate representations in which an interpretation of the text can be constructed before it is committed to graph structure.

---

## 3. Core Proposal: Progressive Semantic Compilation

The proposed system treats text graphicalization as a sequence of semantic transformations:

$$
D
\rightarrow
C
\rightarrow
S
\rightarrow
N
\rightarrow
P
\rightarrow
T
\rightarrow
G.
$$

Here:

* \(D\) is the source document;
* \(C\) is a collection of coherent document chunks;
* \(S\) contains information-preserving summaries;
* \(N\) contains ontology-conditioned normalized text;
* \(P\) contains atomic propositions;
* \(T\) contains structured triples;
* \(G\) is the integrated document graph.

Each transformation has a narrower task than direct graph generation. Information is progressively compressed and regularized until graph construction becomes a constrained mapping problem.

The architecture can therefore be understood as a **semantic compilation pipeline**. Natural language is the source representation. The ontology-grounded graph is the target representation. Intermediate forms make the transformation explicit and inspectable.

---

## 4. Stage I: Structural and Semantic Segmentation

Long documents should first be divided into processing units:

$$
D \rightarrow \{C_1,C_2,\ldots,C_m\}.
$$

For scientific articles, document structure provides useful initial boundaries. Sections, subsections, paragraphs, tables, captions, and similar units can provide structural signals. The framework does not require each structural unit to correspond to exactly one processing chunk. A long section can be subdivided, while several short paragraphs can be grouped when they form a coherent semantic unit.

The purpose of segmentation is narrowly defined:

> Identify the smallest units that contain enough context to support a coherent interpretation while remaining manageable for subsequent transformations.

This stage should avoid ontology assignment or relation extraction. Its output determines what text should be interpreted together.

Different document genres can use different segmentation rules. Scientific papers may rely heavily on headings and rhetorical structure, while books might use chapters, scenes, or narrative transitions.

---

## 5. Stage II: Information Selection and Summarization

Each chunk is next transformed into a compressed representation:

$$
C_i \rightarrow S_i.
$$

Two forms of summarization are proposed.

### 5.1 Generic summarization

Generic summarization attempts to preserve the information judged most significant in the source chunk without reference to a specific ontology.

This representation can be useful when the same document will subsequently be analyzed through several ontologies.

### 5.2 Task-conditioned summarization

Alternatively, summarization can be governed by an explicit preservation specification \(Q\):

$$
S_i = f(C_i,Q).
$$

For scientific literature, \(Q\) might instruct the system to retain information concerning:

* research questions and hypotheses;
* studied entities or populations;
* experimental or observational procedures;
* interventions and comparisons;
* variables and measurements;
* reported relationships;
* findings;
* qualifications and uncertainty;
* limitations.

These categories are part of the proposed design rather than a universal taxonomy of scientific information.

The preservation specification provides a useful separation between **what information should survive compression** and **how that information should subsequently be represented in the ontology**.

---

## 6. Hierarchical Summarization

Long documents may require more than one level of summarization.

A scientific article could, for example, be represented as:

$$
C_{1,1},C_{1,2},\ldots \rightarrow S_1
$$

for one section,

$$
C_{2,1},C_{2,2},\ldots \rightarrow S_2
$$

for another section, followed by

$$
S_1,S_2,\ldots,S_k \rightarrow S_D
$$

for a document-level summary.

This produces a hierarchy:

$$
\text{chunks}
\rightarrow
\text{section representations}
\rightarrow
\text{document representation}.
$$

The highest-level summary should not necessarily replace lower-level representations during graph construction. Repeated summarization can remove detail. A preferable design may retain local representations for extraction while using higher-level summaries to supply global context.

For example, document-level context could help resolve whether two locally mentioned entities refer to the same object without forcing all extraction to operate only on the document summary.

---

## 7. Stage III: Ontology-Conditioned Semantic Normalization

The central transformation is:

$$
N_i=g(S_i,O),
$$

where \(O\) is the ontology.

The objective is to rewrite the retained information using the ontology's conceptual distinctions wherever those distinctions apply.

This operation is deliberately different from ordinary summarization.

Suppose a source states:

> The researchers exposed mice to compound A, after which inflammatory markers decreased.

If the ontology contains the concepts *ExperimentalSubject*, *Intervention*, *ChemicalCompound*, *Biomarker*, and *Decrease*, the normalized representation might resemble:

> ExperimentalSubjects are mice.
> ChemicalCompound A is used as an Intervention on the ExperimentalSubjects.
> Biomarkers associated with inflammation undergo a Decrease after the Intervention.

The exact controlled language remains a design question. What matters is that normalization reduces lexical and syntactic variability while retaining the semantic content needed by the ontology.

OWL provides a formal distinction among classes, properties, individuals, and data values, and its ontology model explicitly supports assertions involving individuals and their relationships. This distinction motivates keeping ontology concepts separate from document-specific instances in the proposed representation.

For example, *Mouse* may be an ontology class while *experimental_group_1* is an instance described by the paper. Similarly, *Treatment* can be an ontology category while *compound_A_administration_1* denotes a particular intervention described in the document.

---

## 8. The Ontology as a Conditioning Language

The ontology can influence more than one stage of the pipeline.

At the summarization stage, it can provide **weak semantic conditioning**. The model is informed about broad types of information that matter but remains free to preserve information that does not yet map cleanly to a particular ontology element.

At the normalization stage, the ontology provides **strong semantic conditioning**. Concepts and relationships should be explicitly mapped to the ontology wherever justified.

This distinction reduces the risk that information will disappear simply because it does not immediately correspond to an ontology term.

Formally, summarization can be represented as:

$$
S_i=f(C_i,Q,O^{weak}),
$$

followed by:

$$
N_i=g(S_i,O^{strong}).
$$

An empirical comparison between no ontology conditioning, weak conditioning, and strong conditioning at the summarization stage would form one component of the evaluation programme.

---

## 9. Stage IV: Proposition Decomposition

Even normalized sentences can express several propositions.

For example:

> Treatment A reduced inflammatory activity and increased survival in mice.

can be decomposed into propositions such as:

$$
p_1:\ \text{Treatment A reduces inflammatory activity}
$$

$$
p_2:\ \text{Treatment A increases survival}
$$

$$
p_3:\ \text{The experimental subjects are mice}.
$$

The proposed transformation is therefore:

$$
N_i \rightarrow P_i
=
\{p_{i1},p_{i2},\ldots,p_{ik}\}.
$$

This stage creates an atomic semantic representation before graph encoding.

The proposition representation can also retain epistemic information. A scientific statement such as

> The results suggest that A may increase B

should not be normalized into an unconditional assertion

$$
A \rightarrow B.
$$

Instead, proposition decomposition can preserve information such as assertion strength, source, modality, uncertainty, experimental context, and attribution.

---

## 10. Stage V: Triple Construction

Atomic propositions are mapped into graph-compatible structures:

$$
P_i \rightarrow T_i.
$$

The fundamental representation is:

$$
(s,r,o),
$$

where \(s\) is a subject, \(r\) a predicate or relation, and \(o\) an object.

This corresponds naturally to the RDF graph model, in which an RDF graph consists of subject-predicate-object triples and predicates represent relationships between resources.

For example:

```text
compound_A_1     instanceOf      ChemicalCompound
mouse_group_1    instanceOf      ExperimentalSubject
treatment_1      uses             compound_A_1
treatment_1      appliedTo        mouse_group_1
treatment_1      causesDecrease   inflammation_marker_1
```

The graph should distinguish ontology-level concepts from textual instances. This avoids forcing every subject and object to correspond directly to a class in the ontology.

The ontology can additionally constrain which predicates are permissible for particular subject and object types. OWL supports class and property definitions and distinguishes object properties linking individuals from datatype properties linking individuals to data values.

Triple construction can therefore become a constrained operation rather than open-ended relation generation.

---

## 11. Stage VI: Graph Integration

Each local unit generates only part of the final graph:

$$
G_i = \text{Graph}(T_i).
$$

The document graph is constructed through an integration operation:

$$
G_D=\bigoplus_{i=1}^{m}G_i.
$$

The integration stage must address several distinct problems:

* entity resolution across chunks;
* duplicate assertions;
* relations spanning multiple chunks;
* temporal or experimental context;
* provenance;
* inconsistent assertions;
* varying levels of specificity;
* references such as pronouns, abbreviations, and aliases.

The operator \(\oplus\) should therefore not be interpreted as simple graph union. It represents a graph reconciliation procedure.

Every graph assertion should ideally retain provenance linking it to the proposition, normalized representation, summary, chunk, and ultimately the source document from which it originated.

This creates a trace such as:

$$
\text{Graph edge}
\rightarrow
\text{Triple}
\rightarrow
\text{Atomic proposition}
\rightarrow
\text{Normalized sentence}
\rightarrow
\text{Source span}.
$$

Such provenance makes individual graph assertions inspectable and provides the basis for error analysis.

---

## 12. Local and Global Processing

The framework naturally separates local extraction from global interpretation.

Local processing handles detailed propositions close to their source text:

$$
C_i
\rightarrow
S_i
\rightarrow
N_i
\rightarrow
P_i
\rightarrow
T_i.
$$

Global processing operates across local outputs:

$$
T_1,\ldots,T_m
\rightarrow
G_D.
$$

Document-level summaries can provide additional context to the integration stage:

$$
S_D
+
\{T_1,\ldots,T_m\}
\rightarrow
G_D.
$$

This permits the system to retain local detail while still reasoning about document-level structure.

---

## 13. Information Compression as a Design Principle

The complete pipeline can be viewed as a sequence of decreasing representational freedom:

$$
\text{raw text}
>
\text{summary}
>
\text{normalized semantic text}
>
\text{atomic propositions}
>
\text{triples}.
$$

At each stage, linguistic variability decreases while structural explicitness increases.

The goal is not maximal compression. The objective is **controlled semantic compression**: reducing representational complexity while preserving information relevant to the target task.

This interpretation provides a useful way of studying failures. Information can disappear at several identifiable boundaries:

$$
C \rightarrow S
$$

may lose a relevant statement;

$$
S \rightarrow N
$$

may incorrectly map a statement to the ontology;

$$
N \rightarrow P
$$

may alter its logical or epistemic structure;

$$
P \rightarrow T
$$

may assign an incorrect relation;

$$
T \rightarrow G
$$

may incorrectly merge entities or assertions.

Direct graph extraction collapses these operations into a much smaller number of observable steps. The proposed framework makes them individually inspectable.

---

## 14. Example

Consider the following simplified scientific passage:

> Mice receiving Compound X showed lower concentrations of Marker Y than untreated controls. The authors suggest that Compound X may suppress the inflammatory pathway associated with Marker Y.

A task-conditioned summary might be:

> Compound X treatment in mice is associated with lower Marker Y concentrations than the untreated condition. The authors propose that Compound X may suppress an inflammatory pathway associated with Marker Y.

An ontology-normalized representation might be:

> Mouse subjects receive an Intervention involving ChemicalCompound X.
> The Intervention is associated with a Decrease in Biomarker Y relative to a ControlCondition.
> The authors propose a possible InhibitoryEffect of ChemicalCompound X on an InflammatoryPathway associated with Biomarker Y.

Proposition decomposition could produce:

```text
Mouse subjects receive Compound X as an intervention.

Marker Y is a biomarker.

Marker Y is lower in the intervention condition than in the control condition.

The authors propose that Compound X may inhibit an inflammatory pathway.

The inflammatory pathway is associated with Marker Y.
```

The graph layer might then contain structures conceptually equivalent to:

```text
compound_X          instanceOf        ChemicalCompound
marker_Y            instanceOf        Biomarker
mouse_group         instanceOf        ExperimentalSubject
treatment_X         instanceOf        Intervention
treatment_X         uses              compound_X
treatment_X         appliedTo         mouse_group
marker_Y            lowerThanIn       control_condition
compound_X          possibleInhibits  pathway_1
pathway_1           associatedWith    marker_Y
```

The exact vocabulary would be determined by the ontology.

The phrase *possibleInhibits* illustrates a broader requirement: epistemic qualification must either be represented in the relation system or reified as additional graph structure. The framework should not convert a qualified claim into a stronger assertion simply to fit it into a triple.

---

## 15. Evaluation Framework

The architecture enables evaluation at several levels rather than relying only on final graph accuracy.

### Segmentation quality

Does each chunk contain sufficient context to interpret the information it contains?

### Preservation quality

For ontology-relevant propositions present in the source, what fraction survive summarization?

A recall-oriented measure could be defined as:

$$
R_S=
\frac{
|\text{relevant propositions preserved in }S|
}{
|\text{relevant propositions in }C|
}.
$$

### Normalization accuracy

Does the canonicalized text preserve the meaning of the summary while correctly applying ontology concepts?

This can be decomposed into concept alignment and semantic faithfulness.

### Proposition accuracy

Does proposition decomposition preserve individual claims, argument structure, negation, qualification, attribution, and uncertainty?

### Triple accuracy

Triples can be evaluated at the subject, predicate, and object level as well as through exact triple matching.

### Graph quality

The integrated graph can be evaluated for entity resolution, duplicate removal, ontology compliance, provenance integrity, and recovery of reference graph structure.

---

## 16. Experimental Comparisons

The central empirical comparison should test the proposed pipeline against simpler alternatives.

A baseline could use direct ontology concept classification followed by pairwise relation classification.

A second baseline could ask a generative model to produce triples directly from source text:

$$
C \rightarrow T.
$$

A third could generate triples directly while supplying the ontology:

$$
(C,O)\rightarrow T.
$$

The proposed approach is:

$$
C
\rightarrow
S
\rightarrow
N
\rightarrow
P
\rightarrow
T.
$$

Ablation experiments can remove individual stages:

$$
C\rightarrow N\rightarrow P\rightarrow T
$$

to test summarization;

$$
C\rightarrow S\rightarrow P\rightarrow T
$$

to test ontology normalization;

and

$$
C\rightarrow S\rightarrow N\rightarrow T
$$

to test explicit proposition decomposition.

The relevant question is therefore not simply whether the final architecture performs well. It is whether each additional representation contributes measurable value.

---

## 17. Key Research Hypotheses

The framework produces several testable hypotheses.

**H1: Ontology-conditioned semantic normalization improves graph extraction accuracy.**

The prediction is that rewriting text into ontology-aligned language before extraction will reduce ambiguity during concept and relation assignment.

**H2: Explicit proposition decomposition improves relation fidelity.**

The prediction is that separating compound statements into atomic claims before triple construction will reduce relation omissions and argument-assignment errors.

**H3: Task-conditioned summarization improves preservation of graph-relevant information compared with generic summarization.**

The expected effect is specifically on information relevant to the target representation.

**H4: Hierarchical processing improves long-document graph construction.**

The prediction is that local extraction combined with document-level integration will retain more detailed information than extraction performed only from a highly compressed document summary.

**H5: Intermediate representations improve error localization.**

Because each transformation is observable, final graph errors can be traced to specific stages of the pipeline.

**H6: Ontology constraints improve structural validity.**

The prediction is that constraining node types and relation types against an ontology will reduce structurally incompatible graph assertions.

These are research hypotheses and require empirical validation.

---

## 18. Ontology Design and Modularity

The framework does not require one universal ontology.

A document may instead be interpreted through several ontology modules. For scientific literature, possible modules could represent experimental design, biological entities, interventions, measurements, causal claims, statistical evidence, or methodological procedures.

The same underlying text could consequently generate different graphs depending on the analytical ontology.

Formally:

$$
G_O=h(D,O).
$$

For two ontologies,

$$
O_a \neq O_b,
$$

it is legitimate that

$$
G_{O_a}\neq G_{O_b}.
$$

The graph should therefore be understood as an ontology-conditioned representation of the document rather than as a unique representation of everything contained in the text.

---

## 19. Human Inspectability

One practical advantage of the proposed architecture is the presence of intermediate representations that can be inspected.

A user can examine:

```text
Source text
    ↓
Summary
    ↓
Ontology-normalized text
    ↓
Atomic propositions
    ↓
Triples
    ↓
Integrated graph
```

When a graph edge appears incorrect, the evaluator can determine whether the problem originated during information selection, ontology alignment, proposition decomposition, triple generation, or graph integration.

This trace can also support human correction. A researcher might approve or modify normalized propositions before graph construction, or inspect only graph assertions with low confidence.

---

## 20. Scientific Literature as the Initial Domain

Scientific documents provide a suitable initial setting for the proposed framework because they contain explicit structural units and because the target analysis can be defined through domain and methodological ontologies.

Recent scientific information-extraction research continues to treat entity identification and relation extraction from full scientific documents as a distinct technical problem; for example, the SciNLP benchmark provides manually annotated entities and relations across full NLP papers.

The proposed framework approaches the problem from a different direction. Its main object of study is the sequence of semantic representations connecting document text to graph structure.

A scientific paper could ultimately generate a graph containing document-specific instances of concepts such as:

```text
ResearchQuestion
Hypothesis
Population
Method
Intervention
Measurement
Variable
Finding
Association
CausalClaim
Limitation
```

together with ontology-defined relationships among them.

The exact ontology should be specified independently of the extraction method and validated as part of the target application.

---

## 21. Open Design Questions

Several questions remain open and should form part of the research programme.

The first concerns the degree of control imposed on the normalized language. It could remain close to ordinary natural language, adopt a tightly constrained grammar, or become an explicitly structured intermediate representation.

The second concerns when ontology information should enter the process. Early ontology conditioning may improve relevance but may also cause information outside the current ontology to disappear. Later conditioning provides greater freedom but leaves more ambiguity for downstream mapping.

The third concerns ontology scale. Supplying a complete large ontology during every normalization step may be inefficient. A future implementation could retrieve a relevant ontology subgraph for each chunk before normalization.

The fourth concerns uncertainty and epistemic status. Scientific literature routinely distinguishes observations, hypotheses, possibilities, author interpretations, previous findings, and established assumptions. The graph representation must preserve these distinctions rather than flattening them into equally asserted edges.

The fifth concerns conflicting evidence. A document graph, and especially a multi-document graph, must be capable of representing disagreement rather than resolving every contradiction into a single assertion.

The sixth concerns provenance. The architecture should determine how much source information must accompany each graph statement to support verification and later reinterpretation.

---

## 22. Proposed System Architecture

The resulting architecture can be summarized as:

```text
                           ┌───────────────┐
                           │   Ontology    │
                           └───────┬───────┘
                                   │
                       conditioning│constraints
                                   │
                                   ▼
                                Document
                                   │
                                   ▼
                                Structural / semantic segmentation
                                   │
                                   ▼
                                Coherent text chunks
                                   │
                                   ▼
                                Information-preserving summarization
                                   │
                                   ▼
                                Ontology-conditioned semantic normalization
                                   │
                                   ▼
                                Atomic proposition decomposition
                                   │
                                   ▼
                                Constrained triple construction
                                   │
                                   ▼
                                Local graphs
                                   │
                                   ├──────── document-level context
                                   │
                                   ▼
                                Entity resolution and graph integration
                                   │
                                   ▼
                                Ontology-grounded document graph
```

A parallel provenance path should retain mappings from every final graph assertion back to its supporting source span.

---

## 23. Research Programme

Development can proceed incrementally.

The first phase should operate on relatively short scientific passages and compare direct triple extraction against ontology normalization followed by triple extraction.

The second phase should introduce proposition decomposition and evaluate whether it improves relation accuracy and preservation of semantic qualifiers.

The third phase should introduce task-conditioned summarization and measure information loss explicitly.

The fourth phase should move from passages to complete scientific papers using structural segmentation and local graph integration.

The fifth phase should address entity resolution, provenance, conflicting claims, and cross-document graph construction.

This sequence allows each proposed mechanism to be tested independently before the complete system is evaluated.

---

## 24. Conclusion

This white paper proposes a shift in the formulation of ontology-based text graphicalization.

Rather than interpreting graph extraction as a large collection of independent questions about whether concepts and relations occur in text, the proposed framework progressively transforms textual information into increasingly constrained semantic representations.

The central pipeline is:

$$
\text{Document}
\rightarrow
\text{Segmentation}
\rightarrow
\text{Summarization}
\rightarrow
\text{Ontology Normalization}
\rightarrow
\text{Atomic Propositions}
\rightarrow
\text{Triples}
\rightarrow
\text{Graph Integration}
}
$$

The ontology performs two roles. It defines the conceptual space in which information will ultimately be represented, and it constrains the transformation from natural-language meaning to graph structure.

The principal methodological idea is **progressive semantic compression**. Each stage reduces representational freedom while attempting to preserve the information required by the next stage. The resulting intermediate representations provide explicit locations at which information preservation, ontology alignment, semantic fidelity, and graph construction can be measured.

The resulting research problem is therefore broader than asking whether a language model can extract a knowledge graph from text. It asks whether long-form natural language can be compiled through a sequence of controlled semantic representations into an ontology-grounded graph while preserving meaning, qualification, provenance, and document-level context.

### References

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* 2018.

Duan, D., Peng, J., Zhang, Y., & Zhang, C. *SciNLP: A Domain-Specific Benchmark for Full-Text Scientific Entity and Relation Extraction in NLP.* EMNLP, 2025.

W3C. *OWL 2 Web Ontology Language: Structural Specification and Functional-Style Syntax, Second Edition.* W3C Recommendation, 2012.

W3C. *OWL 2 Web Ontology Language Primer, Second Edition.* W3C Recommendation, 2012.

W3C. *RDF 1.1 Concepts and Abstract Syntax.* W3C Recommendation, 2014.
