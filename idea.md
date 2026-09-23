# Ontology-Guided Text Graphicalization

## A Hierarchical Semantic Compilation Framework for Scientific Documents

### Abstract

This white paper proposes a framework for transforming long-form scientific text into structured, ontology-grounded knowledge graphs. The central idea is to avoid extracting graph nodes and relations directly from unconstrained text. Instead, the system progressively transforms a document through a sequence of increasingly structured representations: document segmentation, information-preserving summarization, ontology-conditioned semantic normalization, proposition decomposition, triple construction, and graph integration.

The proposed architecture treats an ontology as more than a set of labels to be independently detected. The ontology provides a conceptual vocabulary through which relevant information is expressed and eventually represented as a graph. This creates an intermediate semantic layer between natural language and symbolic graph structure.

The framework is designed around three objectives: preservation of relevant information, explicit control over the conceptual vocabulary used to describe that information, and traceability from graph assertions back to the source text. It is intended particularly for scientific documents, where document structure, specialized terminology, methodological descriptions, and relations among concepts create difficulties for direct entity and relation extraction.

The principal research hypothesis is that graph construction can become more reliable and interpretable when it is treated as a process of progressive semantic compression rather than as a collection of independent classification decisions.

---

## Current Implementation Status

The framework described here has a working reference implementation in this
repository (status checked 2026-09-23). The implementation is provider-neutral
and exposes a scikit-learn-compatible `SemanticGraphicalizer`. Given an
ontology configuration, a prompt configuration, and complete document strings,
it returns one validated NetworkX `MultiDiGraph` per document. When no model
client is supplied, it uses OpenAI structured JSON output with
`gpt-4.1-mini`; deterministic fake clients and callable clients can be
injected for tests or alternative providers.

The implemented execution path is:

```text
Document
  → paragraph/window segmentation
  → per-chunk summarization
  → ontology-conditioned normalization
  → atomic assertion decomposition
  → typed entity and reified-relation extraction
  → document-level cross-chunk/higher-order relation resolution
  → ontology validation and graph materialization
```

The following parts of the proposal are implemented:

* YAML-backed ontology and prompt loading, including ontology terms,
  argument roles, relation schemas, allowed argument types, cardinalities, and
  optional binary projections;
* chunk, summary, normalized-text, entity, argument, relation, and document
  trace data structures;
* paragraph-aware segmentation with character limits and optional overlap;
* strict structured-output schemas for all five model stages:
  `summarize`, `normalize`, `decompose`, `extract`, and `resolve`;
* stable document IDs, conservative entity IDs, model-output validation,
  retry with exponential backoff for transient provider failures, and progress
  statistics for every stage;
* recursive reified graph entities: atomic entities have `relation=None`,
  while relation instances are also nodes whose outgoing edges carry named
  argument roles. Relation arguments may target other relation nodes;
* ontology-constrained graph validation, explicit binary-relation projection,
  and JSON-safe NetworkX node-link serialization;
* provenance on extracted entities, relations, and arguments, including
  document/chunk identifiers, source text, and aligned character spans when
  alignment is possible;
* dynamic D3, static SVG, and indented text renderers, with source mentions,
  relation labels, argument roles, temporal/causal styling, and optional
  timeline layout controls;
* a reusable Project Gutenberg Aesop loader with raw-text and parsed-story
  caching, together with a notebook workflow for loading, graphing, and
  visualizing complete fables.

The implementation is intentionally narrower than the full research proposal.
It currently processes chunks independently before one document-level resolve
call; it does not yet implement hierarchical section/document summarization,
ontology-subgraph retrieval, learned entity/coreference resolution, systematic
duplicate-assertion reconciliation, contradiction handling, RDF/OWL export,
or the proposed empirical benchmark and ablation programme. The default
resolver creates stable IDs from normalized mentions, so semantic aliases and
cross-document identity remain open research problems. The canonical graph is
a reified NetworkX graph; direct binary edges are an explicitly derived view,
not the primary representation.

The current automated test suite covers configuration validation, staged
pipeline behavior, recursive relations, graph validation and round trips,
model-client integration, visualization, and the Aesop cache loader.

## Recursive Reified Semantic Representation

The semantic core of the implementation is a fully recursive reified graph.
It is designed to represent narrative text, scientific discourse, causal and
temporal structure, evidence, logic, measurements, events, states, and
arbitrary n-ary relations through one uniform mechanism.

### One universal semantic primitive

Every semantic object is an `Entity`. This includes ordinary domain objects:

```text
fox, rabbit, protein_X, experiment_17, paper_B, temperature
```

and assertions or relations such as:

```text
the fox chases the rabbit
event_A causes event_B
experiment_17 supports claim_C
paper_B contradicts claim_D
```

The canonical semantic record is:

```python
@dataclass(frozen=True)
class Entity:
    id: str
    type: str
    relation: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
```

The fields have distinct meanings:

* `id` is the stable identity of the semantic object;
* `type` is its ontology-controlled semantic type;
* `relation` is the ontology-controlled relation it instantiates, or `None`
  for an atomic entity;
* `attributes` contains non-structural metadata such as provenance,
  confidence, qualification, source text, modality, attribution, mentions,
  aligned spans, statistical information, and model metadata.

`Entity.type` and `Entity.relation` are independent. For example, both a
`Hypothesis` and a `Conclusion` may instantiate `causes`, while an `Event`
may instantiate `eats`. The type describes what kind of semantic object the
assertion is; the relation describes the relation represented by that object.

### Atomic and relational entities

The only structural distinction is:

```text
Atomic Entity:     relation = None
Relational Entity: relation != None
```

An atomic entity such as `fox` has no argument edges. A relational entity such
as “the fox chases the rabbit” is itself a graph node with its own identity,
type, relation, attributes, and argument edges. Events, states, claims,
measurements, evidence statements, and temporal assertions are not separate
graph primitives; they are ordinary relational entities whose ontology type
and relation express their meaning.

### Reify every relation

Every extracted semantic relation is reified. For:

```text
The fox chases the rabbit.
```

the canonical graph contains:

```text
fox       : type = Animal, relation = None
rabbit    : type = Animal, relation = None
R1        : type = Event,  relation = chases

R1 --agent----> fox
R1 --patient--> rabbit
```

The direct view `fox --chases--> rabbit` is only a derived projection. It is
not the canonical representation because it loses assertion identity,
argument-role detail, provenance, qualification, and the ability to use the
assertion as an argument of another assertion.

Propositions and triples may still be useful names for intermediate pipeline
outputs or evaluation units, but they are not competing structural graph
models. In the implementation, `Summary`, `NormalizedText`, and decomposed
assertion records are transient stage representations; `RelationInstance` and
`Argument` are extraction-time adapters; the integrated graph is always built
from the common `Entity` node model.

### Named arguments, arbitrary arity, and repeated roles

Relations do not have a universal subject/object shape and are not limited to
binary edges. Every argument is a named edge from the relational entity to an
entity:

```text
RelationalEntity --ArgumentRole--> Entity
```

Examples include:

```text
R1 : type = Event, relation = gives
R1 --giver-----> John
R1 --recipient-> Mary
R1 --theme-----> Book
```

```text
R2 : type = Transaction, relation = purchase
R2 --buyer----> Alice
R2 --seller---> Shop
R2 --item-----> Book
R2 --price----> Price20
R2 --time-----> T1
```

```text
R3 : type = Measurement, relation = measures
R3 --sample----> Sample17
R3 --quantity--> Temperature
R3 --value-----> Value42
R3 --unit------> Celsius
R3 --method----> Method3
```

Argument storage is list-like through `Argument` records and
`MultiDiGraph` edges, so a role may repeat:

```text
R4 : relation = supports
R4 --evidence--> experiment_A
R4 --evidence--> experiment_B
R4 --evidence--> dataset_C
R4 --claim-----> claim_X
```

There is therefore no global relation arity. The ontology may define a
relation-specific schema, but unary, binary, ternary, and arbitrary n-ary
relations all use the same topology.

### Three independent controlled vocabularies

The ontology keeps three logically separate vocabularies:

```text
Entity types   → Animal, Event, Experiment, Claim, Measurement, ...
Relations      → chases, causes, supports, contradicts, measures, ...
Argument roles → agent, patient, cause, effect, evidence, claim, ...
```

Relation names and argument-role names must not be collapsed. The relation
describes the relational entity as a whole; the role describes the
participation of one endpoint in that relation. Semantic roles are preferred
to grammatical labels: `giver`, `recipient`, and `theme` are more useful than
`subject`, `object`, and `object2`.

### Ontology-constrained relation schemas

A relation may declare expected argument roles, cardinalities, and optional
target-type constraints:

```yaml
relations:
  supports:
    arguments:
      evidence:
        cardinality: 1..n
        allowed_types: [Experiment, Observation, Dataset, Analysis]
      claim:
        cardinality: 1
        allowed_types: [Claim, Hypothesis, Conclusion, CausalClaim]
```

Schemas are optional, so the representation remains domain-independent. When
a schema is present, graph validation checks required roles, cardinalities,
declared roles, and allowed target types. The current implementation performs
these checks before returning the canonical graph.

### Unrestricted recursive reification

Every argument endpoint is an entity, and a relational entity is also an
entity. Relations can therefore refer to relations without introducing a new
graph mechanism at each level:

```text
R1 : type = Event, relation = increases
R1 --driver----> temperature
R1 --outcome---> reaction_rate

R2 : type = CausalClaim, relation = causes
R2 --cause-----> R1
R2 --effect----> R3

R4 : type = EvidenceStatement, relation = supports
R4 --evidence--> experiment_17
R4 --claim-----> R2
```

There is no semantic `depth`, `level`, or `reification_level` field. Recursion
is expressed by graph topology. Nested assertions may be arbitrarily deep,
and cycles are allowed where they are semantically valid. The implementation
does not impose a global recursive depth or arity limit.

### One mechanism for narrative and scientific semantics

Temporal, causal, evidential, logical, state, and measurement relations are
not structurally special. They are all relational entities:

```text
R1 : type = State, relation = has_state
R1 --bearer----> fox
R1 --state------> hungry

R2 : type = TemporalAssertion, relation = starts_at
R2 --state------> R1
R2 --boundary---> event_3

R3 : type = EvidenceStatement, relation = supports
R3 --evidence---> experiment_17
R3 --claim-------> R4
```

Likewise, a scientific statement can use exactly the same structure:

```text
R4 : type = CausalClaim, relation = causes
R4 --cause------> increased_temperature
R4 --effect-----> faster_reaction_rate
```

The relation `causes` represents the semantic assertion; it does not imply
that the assertion is certainly true. Epistemic type, confidence,
qualification, modality, attribution, evidence, and provenance remain
separate attributes or separate relational entities. A support relation is
itself an assertion and therefore receives its own identity and provenance.

### Canonical NetworkX topology

The canonical graph is a `networkx.MultiDiGraph` whose nodes are entities and
whose structural argument edges use a consistent convention:

```python
graph.add_node(
    entity.id,
    id=entity.id,
    type=entity.type,
    relation=entity.relation,
    attributes=entity.attributes,
)

graph.add_edge(
    relation_entity_id,
    argument_entity_id,
    edge_type="argument",
    role="evidence",
    attributes=argument.attributes,
)
```

Relation names such as `chases` and `causes` are stored on the relational
entity's `relation` field, not used as canonical participant-to-participant
edge names. Argument edges may carry their own attributes, including
confidence, ordering, weights, surface grammatical roles, source spans, and
provenance.

### Identity and provenance at every level

Relational entities are not silently deduplicated merely because they share a
relation and arguments. Two documents may independently assert the same
relation, and those assertions may need to remain distinct:

```text
R1 : relation = causes, cause = A, effect = B, provenance = paper_1
R2 : relation = causes, cause = A, effect = B, provenance = paper_2
```

The current pipeline preserves independent relation IDs and provenance during
recursive graph materialization. Entity mention resolution may merge atomic
entities when their stable resolver IDs match, but semantic aliasing and
assertion reconciliation remain explicit later-stage research problems.

Each entity and argument edge preserves, where available:

```text
document_id, chunk_id, source_text,
start_char, end_char, mention spans,
confidence, qualification, modality, attribution
```

Provenance is not inherited from a lower-order relation as though it were the
provenance of a higher-order assertion. A causal assertion and a support
assertion that refers to it retain separate provenance records.

### Pipeline consequences

The recursive semantic model determines the processing sequence:

```text
1. segment the document;
2. summarize and normalize where useful;
3. decompose normalized text into atomic assertions;
4. identify atomic entities;
5. identify relation instances and assign ontology types;
6. assign named argument roles;
7. resolve argument references, including relation references;
8. create relational entity nodes and argument edges;
9. detect explicitly supported higher-order relations;
10. resolve cross-chunk references at document level;
11. validate against ontology schemas;
12. materialize the unified MultiDiGraph;
13. optionally derive conventional binary projections.
```

The ability to represent a higher-order relation does not authorize the model
to invent one. Relations are extracted only when supported by the source text,
prompt task, and configured ontology.

The extraction contract mirrors the graph model. A model returns atomic
entities plus relation instances whose arguments reference entity or relation
IDs from the same response:

```json
{
  "id": "r18",
  "type": "EvidenceStatement",
  "relation": "supports",
  "arguments": [
    {"role": "evidence", "entity_id": "experiment_17", "attributes": {}},
    {"role": "claim", "entity_id": "r12", "attributes": {}}
  ],
  "attributes": {"confidence": 0.94}
}
```

The prompt supplies the allowed entity types, relation types, argument roles,
and relation-specific schemas. Malformed references, unknown vocabulary
values, and invalid argument structures are rejected before graph integration.

### Validation, projection, and serialization

The canonical graph maintains these invariants:

1. Every semantic node has `id`, `type`, `relation`, and `attributes`.
2. Atomic entities have `relation=None`.
3. Relational entities have a non-null relation and argument edges.
4. Every argument edge has a valid role and points to an existing entity.
5. Relation schemas enforce required roles, cardinalities, declared roles,
   and optional target-type constraints.
6. Repeated roles, n-ary relations, relation-to-relation arguments, and valid
   cycles are supported.
7. Provenance and edge attributes survive graph integration and serialization.

`project_binary_relations` derives ordinary direct edges only when the
ontology declares an explicit ordered two-role projection. N-ary relations
are not forced into binary form. `graph_to_dict` and `graph_from_dict` use a
JSON-safe node-link representation so recursive and cyclic graph topology is
represented by IDs and edges rather than nested Python objects.

Visualization can show the canonical reified view with entity types, relation
names, and argument-role labels, or a simpler projected view when appropriate.
This makes a structure such as `supports(R2, R1)` visible as a `supports`
relation node connected by `evidence` and `claim` edges, with `R1` itself
displayed as the nested `causes` relation node.

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

In the reference implementation, this stage is represented by typed,
reified relation entities rather than by bare triples alone. A relation entity
is a graph node with a configured `type` and `relation`; its argument edges
carry roles such as `actor`, `target`, `cause`, or `effect`. This supports
ternary relations, repeated argument roles, nested relations, provenance, and
qualified or attributed assertions. The helper
`project_binary_relations` can derive direct edges for ontology relations that
declare an explicit two-role projection.

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

The current implementation materializes a canonical `MultiDiGraph` after
chunk-level extraction and a document-level `resolve` stage. It validates
relation types, argument roles, allowed target types, and cardinalities before
returning the graph. Extracted entities and relations retain source text and
document/chunk provenance; character spans are attached when the source text
can be aligned to the original document.

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

## 22. System Architecture: Proposal and Current Implementation

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

In the current implementation, the diagram's “entity resolution and graph
integration” block is realized by conservative mention-based ID resolution,
one document-level `resolve` model call, and deterministic ontology validation
plus materialization. The remaining global reasoning steps in the diagram,
including hierarchical document context and conflict reconciliation, are still
research work.

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
