## 1_58 candidates 4
### 1_58 [17]
- Target: `industry LLMs.`
- Filter: It fails criterion 3 because the target, "industry LLMs," is a very short generic phrase with little conceptual depth, making it hard to create five meaningful, discriminative answer options.
- Probe: According to the paper "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits", post-training quantization is widely used in
- Source: However, post-training quantization is sub-optimal, even though it is widely used in industry LLMs.

### 1_58 [35]
- Target: `0`
- Filter: It fails criterion 3 because the target is the trivially short single symbol '0', which offers little conceptual depth and would make it hard to create five meaningful, discriminative answer choices.
- Probe: According to the paper "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits", the additional value added to the original 1-bit BitNet, resulting in 1.58 bits in the binary system, was
- Source: We have added an additional value of 0 to the original 1-bit BitNet, resulting in 1.58 bits in the binary system.

### 1_58 [58]
- Target: `language tasks`
- Filter: It is incorrectly formulated (criterion 5): the source says the models were evaluated on a range of language tasks including those benchmarks, not that those benchmarks themselves 'were evaluated as language tasks,' making the probe-answer pair awkward and inaccurate.
- Probe: According to the paper "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits", ARC-Easy, ARC-Challenge, Hellaswag, Winogrande, PIQA, OpenbookQA, and BoolQ were evaluated as
- Source: We evaluated the zero-shot performance on a range of language tasks, including ARC-Easy~\cite{arc}, ARC-Challenge~\cite{arc}, Hellaswag~\cite{hellaswag}, Winogrande~\cite{winoGrande}, PIQA~\cite{piqa}, OpenbookQA~\cite{openbookqa}, and BoolQ~\cite{boolq}.

### 1_58 [92]
- Target: `the cost from other components`
- Filter: It is unsuitable under criterion 1 because the probe is essentially a near-verbatim restatement of the source sentence with only the final generic phrase omitted, making the answer a tautological completion rather than a conceptually distinct MCQA target.
- Probe: According to the paper "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits", in the energy-consumption comparison between BitNet b1.58 and the FP16 LLaMA LLM baseline, the quantity that becomes smaller for larger models as the percentage of \emph{nn.Linear} grows with model size is
- Source: This is due to the fact that the percentage of \emph{nn.Linear} grows with the model size, while the cost from other components is smaller for larger models.

## BOFT candidates 18
### BOFT [130]
- Target: `another $d$ nodes.`
- Filter: It fails criterion 3 because the target is a trivially short, obvious phrase ('another d nodes') that offers little conceptual depth and would make constructing five meaningful distractors unnatural.
- Probe: According to the paper "Parameter-Efficient Orthogonal Finetuning via Factorization", in the information transmission view, a $d$-dimensional dense square matrix is interpreted as a dense connectivity from $d$ nodes to
- Source: The motivation behind the information transmission view comes from the observation that a $d$-dimensional dense square matrix can be interpreted as a dense connectivity from $d$ nodes to another $d$ nodes.

### BOFT [133]
- Target: `$\bm{B}_i$.`
- Filter: It fails criterion 3 because the target is just the bare notation "$\bm{B}_i$", a trivially short symbol-level answer with little conceptual depth, making it hard to build five meaningful MCQA options.
- Probe: According to the paper "Parameter-Efficient Orthogonal Finetuning via Factorization", when the dense matrix $\bm{R}$ is represented as a product of multiple matrices $\bm{B}_m\bm{B}_{m-1}\cdots\bm{B}_1$, the authors interpret this as sequential information exchange based on the graphs induced by
- Source: Therefore, representing the dense matrix $\bm{R}$ with multiple matrices $\bm{B}_m\bm{B}_{m-1}\cdots\bm{B}_1$ can also be interpreted as performing sequential information exchange based on the graphs induced by $\bm{B}_i, \forall i$.

### BOFT [135]
- Target: `the $(i+1)$-th level nodes.`
- Filter: It fails criterion 1 because the completion is essentially tautological from the probe itself: given 'from the i-th level nodes to', the natural restatement is simply 'the (i+1)-th level nodes,' leaving little substantive knowledge to distinguish among options.
- Probe: According to the paper "Parameter-Efficient Orthogonal Finetuning via Factorization", in the induced graph obtained by unrolling the matrix multiplication for the factorization $\bm{R}=\bm{B}_5\bm{B}_4\bm{B}_3\bm{B}_2\bm{B}_1$, the matrix $\bm{B}_i$ is viewed as the connectivity matrix from the $i$-th level nodes to
- Source: In the induced graph, the matrix $\bm{B}_i$ is viewed as the connectivity matrix from the $i$-th level nodes to the $(i+1)$-th level nodes.

### BOFT [138]
- Target: `node $3$`
- Filter: It fails criterion 3 because the target is merely the label "node 3," a trivially short graph-node identifier with little conceptual depth, making it hard to create meaningful 5-option distractors beyond arbitrary node numbers.
- Probe: According to the paper "Parameter-Efficient Orthogonal Finetuning via Factorization", when considering $\bm{R}=\bm{B}_4\bm{B}_3\bm{B}_2\bm{B}_1$ as the mapping from the source nodes in the first level to the receiver nodes in the $5$-th level, information from node $1$ cannot be transmitted to
- Source: If we only consider $\bm{R}=\bm{B}_4\bm{B}_3\bm{B}_2\bm{B}_1$ which corresponds to the source nodes in the first level and the receiver nodes in $5$-th level, then we find that information from the node $1$ cannot be transmitted to the node $3$.

### BOFT [144]
- Target: `the $(m+1)$-th level.`
- Filter: It is unsuitable under criterion 3 because the target is just a short positional phrase ('the (m+1)-th level'), making the completion too trivial and not conceptually rich enough for five meaningful MCQA options.
- Probe: According to the paper "Parameter-Efficient Orthogonal Finetuning via Factorization", in the information transmission view for the factorization matrices $\bm{B}_m,\cdots,\bm{B}_1$ on a $d\times(m+1)$ grid, if $\bm{R}\in\mathbb{R}^{d\times d}$ is dense, information from every node in the first level must be transmitted to
- Source: Generally, for a matrix $\bm{R}\in\mathbb{R}^{d\times d}$ to be dense, the $m$ factorization matrices $\bm{B}_m,\cdots,\bm{B}_1$ needs to correspond to a set of directed edges on a $d\times (m+1)$ grid where one directed edge can only connect two nodes between adjacent levels (\emph{i.e.}, columns), such that information from every node in the first level can be transmitted to every node in the $(m+1)$-th level.

### BOFT [150]
- Target: `the last level`
- Filter: It is incorrectly formulated (criterion 5) because the source fact says each first-level node must have at least one path to every node in the last level, whereas the target simplifies this to just 'the last level,' losing the essential 'every node' requirement.
- Probe: According to the paper "Parameter-Efficient Orthogonal Finetuning via Factorization", under the information transmission view, the dense connectivity desideratum requires every node in the first level to have at least one path to
- Source: Under the information transmission view, the general desiderata towards our goal are \emph{(i)} \textbf{dense connectivity}: every node in the first level has at least one path to every node in the last level, and \emph{(ii)} \textbf{minimum free edges}: the total number of edges should be as small as possible under the orthogonality constraint.

### BOFT [153]
- Target: `$\pm 1 $.`
- Filter: Unsuitable because the target is a trivially short symbolic answer (criterion 3) and it would be hard to construct five natural, meaningful distractors for 'those entries can only be ±1' (criterion 4).
- Probe: According to the paper "Parameter-Efficient Orthogonal Finetuning via Factorization", in the discussion of orthogonality constraints and counting edges between adjacent levels, for a $d\times d$ orthogonal matrix with $d$ non-zero entries, those entries can only be
- Source: These $d$ edges is necessary for orthogonality and should not be counted into the number of edges, because these elements are not trainable (\emph{e.g.}, for a $d\times d$ orthogonal with $d$ non-zero entries, these entries can only be $\pm 1$).

### BOFT [175]
- Target: `$\tilde{\bm{B}}(d,2)$.`
- Filter: Unsuitable by criterion 1: the probe is essentially a notation-completion task where the target is just the missing last term of the factorization already stated in the question, making it tautological rather than a meaningful conceptual MCQA.
- Probe: According to the paper "Parameter-Efficient Orthogonal Finetuning via Factorization", in the recursive definition of the $d$-dimensional butterfly matrix $\bm{B}(d)$ for $d=2^N$, the last butterfly component appearing in the factorization $\tilde{\bm{B}}(d,d)\tilde{\bm{B}}(d,\frac{d}{2})\cdots$ is
- Source: With $d=2^N$, we then define the $d$-dimensional \emph{butterfly matrix} $\bm{B}(d)\in\mathbb{R}^{d\times d}$ recursively as

\begin{equation}
\footnotesize
\begin{aligned}
    \!\!\bm{B}(d)=\tilde{\bm{B}}(d,d)\!\cdot\!\begin{bmatrix}
    \bm{B}_1(\frac{d}{2})\!\!\!\!\! & \bm{0} \\
    \bm{0}\!\!\!\!\! &\bm{B}_2(\frac{d}{2})
    \end{bmatrix}= \tilde{\bm{B}}(d,d)\tilde{\bm{B}}(d,\frac{d}{2})\cdots\tilde{\bm{B}}(d,2),
\end{aligned}
\end{equation}

where $\bm{B}_1(\frac{d}{2})$ and $\bm{B}_2(\frac{d}{2})$ are two $\frac{d}{2}$-dimensional butterfly matrices.

### BOFT [192]
- Target: `$\bm{I}_d$.`
- Filter: This probe is tautological/circular (criterion 1): once the question states that each butterfly component is orthogonal, the completion follows directly by definition as the identity matrix, leaving little room for meaningful 5-option discrimination.
- Probe: According to the paper "Parameter-Efficient Orthogonal Finetuning via Factorization", in the BOFT forward-pass formulation where each butterfly component $\tilde{\bm{B}}^b_{(d,j)}$ is required to be orthogonal, $\big(\tilde{\bm{B}}^b_{(d,j)}\big)^\top\tilde{\bm{B}}^b_{(d,j)}$ equals
- Source: Combining pieces, the forward pass in BOFT is

\begin{equation}
    \footnotesize
    \begin{aligned}\nonumber
    \bm{z}=\big(\bm{R}(m,b)\cdot\bm{W}^0\big)^\top\bm{x},~~~~\text{s.t.}~\bigg{\{}\bm{R}(m,b)=\prod_{i=1}^m \tilde{\bm{B}}^b_{(d,i)}~~\&~~\underbrace{\big(\tilde{\bm{B}}^b_{(d,j)}\big)^\top\tilde{\bm{B}}^b_{(d,j)}=\tilde{\bm{B}}^b_{(d,j)}\big(\tilde{\bm{B}}^b_{(d,j)}\big)^\top\!=\bm{I}_d}_{\forall j\in [1, m]}\bigg{\}},
    \end{aligned}
\end{equation}

where we denote $\tilde{\bm{B}}^b(d,2^{m - i+1})$ as $\tilde{\bm{B}}^b_{(d,i)}$ for simplicity, and $\bm{I}_d$ is an identity matrix 

### BOFT [195]
- Target: `BOFT($m$,$b$).`
- Filter: It fails criterion 1 because the probe is essentially asking for the exact notation convention introduced in the sentence, so the target is just a near-tautological restatement rather than a concept that supports meaningful distractors.
- Probe: In the paper "Parameter-Efficient Orthogonal Finetuning via Factorization", the notation used for BOFT with $\bm{R}(m,\frac{b}{2})$ when $b\geq 2$ is
- Source: For convenience, we denote BOFT with $\bm{R}(m,\frac{b}{2})$ as BOFT($m$,$b$), where $b\geq 2$.

### BOFT [198]
- Target: `$b$.`
- Filter: Unsuitable under criterion 1 and 3: the completion is just the symbolic parameter already introduced in the probe ('BOFT(1,b)'), making the answer a trivially short variable rather than a concept that could support five meaningful distractors.
- Probe: According to the paper "Parameter-Efficient Orthogonal Finetuning via Factorization", when $m=1$, BOFT reduces to the block-diagonal OFT, and the block size in that case is
- Source: When $\thickmuskip=2mu \medmuskip=2mu m=1$, then BOFT($1$,$b$) reduces to the block-diagonal OFT~\cite{qiu2023controlling} with the block size $b$.

### BOFT [226]
- Target: `$\bm{B}_{d-1,1}(d)\bm{B}^\top_{d-1,2}(d)\cdots\bm{B}_{1,1}(d)\bm{B}^\top_{1,2}(d)$.`
- Filter: It is unsuitable under criterion 4 because the answer is a long, highly specific symbolic product, and constructing five natural, meaningful distractor formulas would be artificial and mostly test verbatim recall of notation order rather than conceptual understanding.
- Probe: According to the paper "Parameter-Efficient Orthogonal Finetuning via Factorization", in the theorem on the expressivity of BOFT, the product of butterfly matrices used to approximate all orthogonal matrices of size $d$ is
- Source: For the butterfly matrix to approximate all orthogonal matrices of size $d$, we can multiply butterfly matrices with  $\bm{B}_{d-1,1}(d)\bm{B}^\top_{d-1,2}(d)\cdots\bm{B}_{1,1}(d)\bm{B}^\top_{1,2}(d)$, where $\bm{B}_{i,j}(d),\forall i,\forall j$ are butterfly matrices.

### BOFT [228]
- Target: `$\bm{R}^G(m_1,b_1,m_2,b_2,l)=\bm{R}_{l,1}(m_1,b_1)\bm{R}_{l,2}^T(m_2,b_2)\cdots\bm{R}_{1,1}(m_1,b_1)\bm{R}_{1,2}^T(m_2,b_2)$.`
- Filter: It fails criterion 4 because the target is a long, highly specific matrix formula, making it unnatural to construct five meaningful and plausibly distinct multiple-choice distractors rather than arbitrary notation variants.
- Probe: In the paper "Parameter-Efficient Orthogonal Finetuning via Factorization", the final orthogonal matrix in the BOFT generalization is
- Source: Theorem~\ref{thm:exp_boft} suggests a simple generalization for BOFT -- the final orthogonal matrix is generalized to $\thickmuskip=2mu \medmuskip=2mu \bm{R}^G(m_1,b_1,m_2,b_2,l)=\bm{R}_{l,1}(m_1,b_1)\bm{R}_{l,2}^T(m_2,b_2)\cdots\bm{R}_{1,1}(m_1,b_1)\bm{R}_{1,2}^T(m_2,b_2)$ where $\bm{R}_{i,j}^T(m,b)$ denotes the orthogonal matrix used in BOFT.

### BOFT [229]
- Target: `the orthogonal matrix used in BOFT.`
- Filter: It fails criterion 3 because the target is a very generic descriptive phrase ('the orthogonal matrix used in BOFT') rather than a distinctive concept or named object that would support five meaningful distractors.
- Probe: In the paper "Parameter-Efficient Orthogonal Finetuning via Factorization", in the generalized BOFT formulation of the final orthogonal matrix $\bm{R}^G(m_1,b_1,m_2,b_2,l)$, $\bm{R}_{i,j}^T(m,b)$ denotes
- Source: Theorem~\ref{thm:exp_boft} suggests a simple generalization for BOFT -- the final orthogonal matrix is generalized to $\thickmuskip=2mu \medmuskip=2mu \bm{R}^G(m_1,b_1,m_2,b_2,l)=\bm{R}_{l,1}(m_1,b_1)\bm{R}_{l,2}^T(m_2,b_2)\cdots\bm{R}_{1,1}(m_1,b_1)\bm{R}_{1,2}^T(m_2,b_2)$ where $\bm{R}_{i,j}^T(m,b)$ denotes the orthogonal matrix used in BOFT.

### BOFT [232]
- Target: `better expressiveness.`
- Filter: Unsuitable because it is both tautological—the target 'better expressiveness' merely restates the probe’s mention of expressiveness (criterion 1)—and incorrectly formulated relative to the source, which says 'despite its universal expressiveness' rather than 'better expressiveness' (criterion 5).
- Probe: In the paper "Parameter-Efficient Orthogonal Finetuning via Factorization", the authors note that although full finetuning has universal expressiveness, this does not always lead to better performance in finetuning despite its
- Source: However, we note that better expressiveness does not always lead to better performance in finetuning, as full finetuning, despite its universal expressiveness, often yields unsatisfactory performance.

### BOFT [241]
- Target: `training stability.`
- Filter: It fails criterion 5 because the source fact says the preservation benefits both training stability and generalization, so the probe's target 'training stability' alone is an incomplete and inaccurately formulated answer.
- Probe: According to the paper "Parameter-Efficient Orthogonal Finetuning via Factorization", in the discussion of OFT and BOFT preserving the largest singular value, that is, the spectral norm of the pretrained weight matrix $\bm{W}^0$ after left-multiplication by an orthogonal matrix $\bm{R}$, this preservation has been shown to greatly benefit
- Source: Such a preservation has been shown to greatly benefit training stability and generalization~\cite{miyato2018spectral,yoshida2017spectral}.

### BOFT [245]
- Target: `$\bm{R}$`
- Filter: It fails criterion 3 because the target is just the single symbolic identifier $\bm{R}$, which is too trivially short and lacks enough conceptual substance to support five meaningful multiple-choice options.
- Probe: According to the paper "Parameter-Efficient Orthogonal Finetuning via Factorization", BOFT is viewed as learning the bilinear similarity matrix
- Source: BOFT can be viewed as learning the bilinear similarity matrix $\bm{R}$ with a strong regularity (\emph{i.e.}, $\bm{R}$ needs to be orthogonal), which intrinsically connects to distance metric learning~\cite{xing2002distance} and bilinear form~\cite{roman2005advanced}.

### BOFT [257]
- Target: `scratch.`
- Filter: It fails criterion 5 because the probe is incorrectly formulated by adding 'use random initialization,' which is not supported by the source fact, and the target 'scratch' is also too trivial to support meaningful 5-way distractors (criterion 3).
- Probe: According to the paper "Parameter-Efficient Orthogonal Finetuning via Factorization", in comparison to butterfly-based sparse training, works that directly reparameterize weight matrices with the butterfly parameterization typically use random initialization when training neural networks from
- Source: They typically focus on reparameterizing the weight matrices directly with the butterfly parameterization and training neural networks from scratch.

## ByteLatent candidates 14
### ByteLatent [6]
- Target: `flop controlled scaling study`
- Filter: Criterion 5: the probe is incompletely formulated relative to the source fact, since the paper says it presents the first flop controlled scaling study of byte-level models, but the target truncates that claim to the vague phrase 'flop controlled scaling study.'
- Probe: According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", the authors present a
- Source: We present the first flop controlled scaling study of byte-level models up to 8B parameters and 4T training bytes.

### ByteLatent [34]
- Target: `{\textsc BLT} models.`
- Filter: Unsuitable because the answer is effectively exposed by the probe itself: the paper title names Byte Latent Transformer, so completing it with BLT models is nearly tautological and violates criterion 2 (and borderline 1).
- Probe: According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", model size and patch size can be simultaneously increased while maintaining the same inference flop budget with
- Source: Finally, with {\textsc BLT} models, we can simultaneously increase model size and patch size  while maintaining the same inference flop budget.

### ByteLatent [40]
- Target: `\{0,1\}.`
- Filter: It is unsuitable because the target is just the binary set {0,1}, making it trivially short and hard to construct five meaningful distractors (criteria 3 and 4).
- Probe: According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", in the formal definition of the patching function \(f_p\), each \(x_i\) is mapped to the set
- Source: Formally, a patching function $f_p$ segments a sequence of bytes $\pmb{x}=\{x_i,|i=1,\ldots n\}$ of length $n$ into a sequence of $m < n$ patches $\pmb{p}=\{p_j|j=1,\ldots,m\}$ by mapping each $x_i$ to the set \{0,1\} where 1 indicates the start of a new patch.

### ByteLatent [41]
- Target: `1.`
- Filter: Unsuitable under criteria 3 and 4: the target is a trivially short binary value ('1'), and it would be unnatural to construct five meaningful distractors for such a yes/no-style indicator choice.
- Probe: According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", in the formal definition of the patching function $f_p$, the binary indicator value that indicates the start of a new patch is
- Source: Formally, a patching function $f_p$ segments a sequence of bytes $\pmb{x}=\{x_i,|i=1,\ldots n\}$ of length $n$ into a sequence of $m < n$ patches $\pmb{p}=\{p_j|j=1,\ldots,m\}$ by mapping each $x_i$ to the set \{0,1\} where 1 indicates the start of a new patch.

### ByteLatent [107]
- Target: `$w_{\mathcal{E}}$.`
- Filter: It is unsuitable because the target is merely a paper-specific variable name ($w_{\mathcal{E}}$) rather than a substantive concept, making it trivially notational and hard to generate five meaningful distractors (criteria 3 and 4).
- Probe: In the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", when describing the transformer layers’ local block causal attention mask, each byte attends to a fixed window of how many preceding bytes, namely
- Source: The transformer layers use a \textit{local block causal} attention mask; each byte attends to a fixed window of $w_{\mathcal{E}}$ preceding bytes that in general can cross the dynamic patch boundaries but can not cross document boundaries.

### ByteLatent [111]
- Target: `the byte $b_i$.`
- Filter: It is effectively tautological and too trivial for a 5-option MCQ, since the probe nearly restates the source sentence and the target is just the simple phrase 'the byte b_i' rather than a distinctive concept or method.
- Probe: According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", in encoder hash n-gram embeddings used to create robust, expressive representations at each step $i$, what is modeled individually in addition to being modeled as part of a byte n-gram is
- Source: In {\textsc BLT}, we achieve this by modeling both the byte $b_i$ individually \textit{and} as part of a byte n-gram.

### ByteLatent [126]
- Target: `patch $j$.`
- Filter: It is unsuitable because the target is a trivially short, index-mirroring phrase ('patch j') that is essentially obvious from the probe’s own notation, making the item too close to tautological and weak for constructing 5 meaningful distractors (criteria 1 and 3).
- Probe: According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", the patch-specific masking strategy for attention defines the bytes that each query $Q_j$ can attend to as those in
- Source: We use a masking strategy specific to patching where each query $Q_j$ only attends to the keys and values that correspond to the bytes in patch $j$.

### ByteLatent [134]
- Target: `$l_{\mathcal{D}} << l_{\mathcal{G}}$.`
- Filter: Unsuitable under criterion 4: the target is just a relative comparison ($l_{\mathcal{D}} \ll l_{\mathcal{G}}$), so constructing 5 natural, meaningful distractor options would be awkward and largely reduce to arbitrary inequality variants.
- Probe: According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", the number of layers in the local decoder $\mathcal{D}$ compared with the number of layers in the global model $\mathcal{G}$ is
- Source: Similar to the local encoder, the local decoder  $\mathcal{D}$ is a lightweight transformer-based model with $l_{\mathcal{D}} << l_{\mathcal{G}}$ layers, that decodes a sequence of global patch representations $o_j$, into raw bytes, $y_i$.

### ByteLatent [151]
- Target: `scaling law experiments`
- Filter: It fails criterion 5 because the probe is incorrectly formulated and grammatically incomplete, so it does not cleanly or accurately express that the former 2-trillion-token dataset is used for scaling law experiments.
- Probe: According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", with the former 2 trillion-token dataset, cleaned and filtered from publicly available sources, conducts
- Source: The former is used for scaling law experiments on optimal number of tokens as determined by~\cite{dubey2024llama} to determine the best architectural choices for {\textsc BLT}, while the latter is used for a complete pre-training run to compare with Llama 3 on downstream tasks.

### ByteLatent [176]
- Target: `the more expensive latent transformer.`
- Filter: It fails criterion 5 (incorrectly formulated): the probe ends with 'packing batches of patches into' but the source says they pack batches of patches 'to avoid padding steps in the more expensive latent transformer,' so the target does not grammatically or semantically complete the statement.
- Probe: According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", for efficiency reasons, the BLT training implementation avoids padding steps by packing batches of patches into
- Source: For efficiency reasons, our implementation of {\textsc BLT} training packs batches of patches to avoid padding steps in the more expensive latent transformer.

### ByteLatent [204]
- Target: `a constant.`
- Filter: It is incorrectly formulated and too trivial: in the actual BPB definition the normalization constant is specifically ln(2), so the target 'a constant' is an imprecise generic phrase that does not properly capture the source fact (criterion 5, also bordering on criterion 3).
- Probe: According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", in the definition of $\text{BPB}(x)$, the uncertainty over the data $\pmb{x}$ measured by the sum of the cross-entropy loss is normalized by the total number of bytes in $\pmb{x}$ and
- Source: Specifically:

\begin{align}
    \text{BPB}(x)&= \frac{\mathcal{L}_{CE}(\pmb{x})}{\ln(2)\cdot n_{\text{bytes}}}
\end{align}

where the uncertainty over the data $\pmb{x}$ as measured by the sum of the cross-entropy loss is normalized by the total number of bytes in $\pmb{x}$ and a constant.

### ByteLatent [283]
- Target: `larger patches.`
- Filter: Unsuitable under criterion 1 because the probe is effectively a near-verbatim sentence completion from the paper ('larger models taking fewer steps on ...'), making the target 'larger patches' a tautological quote completion rather than a conceptually discriminative MCQ answer.
- Probe: In the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", in the fixed inference scaling study used to test whether larger models taking fewer steps can outperform smaller models taking more steps, the authors say the larger models are taking fewer steps on
- Source: We conduct a fixed inference scaling study to test the hypothesis that larger models taking fewer steps on larger patches might perform better than smaller models taking more steps.

### ByteLatent [415]
- Target: `dynamic patching schemes.`
- Filter: It fails criterion 5 because the probe is incorrectly formulated: in the source, 'dynamic patching schemes' is the umbrella category that includes the listed boundary-predictor and entropy-based methods, not an additional item to complete the list.
- Probe: According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", nawrot et al. (2023) further improve the byte-level model with a learned end-to-end boundary-predictor, a tokenizer-supervised boundary-predictor, an entropy-based patching model similar to BLT, and
- Source: \cite{nawrot-etal-2023-efficient} further improve this with the help of dynamic patching schemes, including a boundary-predictor that is learned in an end-to-end fashion, a boundary-predictor supervised using certain tokenizers, as well as an entropy-based patching model similar to {\textsc BLT}, and show that this approach can outperform the vanilla transformers of the time on language modeling tasks at a 40M parameter scale on ~400M tokens.

### ByteLatent [451]
- Target: `50\% reductions.`
- Filter: It fails criterion 5 (incorrectly formulated) because the probe asks only that BLT 'can achieve' and the target '50% reductions' omits the crucial quantity being reduced—'inference flops'—so the question-answer pair does not properly reflect the source fact.
- Probe: According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", in the authors' extensive scaling study, {	extsc BLT} can achieve
- Source: Our extensive scaling study demonstrates that {\textsc BLT} models can match the performance of tokenization-based models  like Llama 3 at scales up to 8B and 4T bytes, and can trade minor losses in evaluation metrics for up to 50\% reductions in inference flops.

## DPO candidates 20
### DPO [11]
- Target: `the LM.`
- Filter: Unsuitable under criterion 3: the target 'the LM' is a trivially short, generic phrase with little conceptual depth, making it hard to create five meaningful discriminative options.
- Probe: In the paper "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", the resulting algorithm, Direct Preference Optimization (DPO), eliminates the need to sample during fine-tuning from
- Source: The resulting algorithm, which we call \textit{Direct Preference Optimization} (DPO), is stable, performant, and computationally lightweight, eliminating the need for  sampling from the LM during fine-tuning or performing significant hyperparameter tuning.

### DPO [56]
- Target: `\textit{relative} human judgments of response quality`
- Filter: Unsuitable because the probe already states nearly the entire answer ('human judgments of response quality') and only asks for the missing adjective 'relative,' making it effectively tautological and too trivial for meaningful 5-option MCQA distractors (criteria 1 and 3).
- Probe: According to the paper "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", the type of human judgments of response quality that are often easier to collect than expert demonstrations, motivating subsequent fine-tuning of LLMs with datasets of human preferences, are
- Source: Despite the success of instruction tuning, \textit{relative} human judgments of response quality are often easier to collect than expert demonstrations, and thus subsequent works have fine-tuned LLMs with datasets of human preferences, improving proficiency in translation \citep{kreutzer-etal-2018-reliability}, summarization \citep{stiennon2022learning,ziegler2020finetuning}, story-telling \citep{ziegler2020finetuning}, and instruction-following \citep{ouyang2022training,ramamurthy2023is}.

### DPO [82]
- Target: `$y_w\succ y_l \mid x$.`
- Filter: It fails criterion 5 because the probe is incompletely formulated—'the authors use the notation' does not specify what the notation refers to, so the answer cannot be uniquely and properly recovered from the question alone.
- Probe: According to the paper "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", the authors use the notation
- Source: These are then presented to human labelers who express preferences for one answer, denoted as $y_w\succ y_l \mid x$ where $y_w$ and $y_l$ denotes the preferred and dispreferred completion amongst $(y_1, y_2)$ respectively.

### DPO [96]
- Target: `$0$.`
- Filter: Unsuitable because the target is a trivially short numeric value ('$0$'), making it hard to construct five meaningful, conceptually distinct distractors beyond arbitrary numbers (criteria 3 and 4).
- Probe: According to the paper "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", to ensure a reward function with lower variance, prior works normalize the rewards so that, for all $x$, \(\mathbb{E}_{x,y\sim \mathcal{D}}\left[r_\phi(x, y)\right]\) takes the value
- Source: To ensure a reward function with lower variance, prior works normalize the rewards, such that  $\mathbb{E}_{x,y\sim \mathcal{D}}\left[r_\phi(x, y)\right] = 0$ for all $x$.

### DPO [117]
- Target: `$\pi_r$.`
- Filter: It is unsuitable under criterion 3 because the target is merely a very short notation symbol ($\pi_r$), making the question a trivial recall of variable naming rather than a concept with enough depth for 5 meaningful options.
- Probe: According to the paper "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", Eq.~\ref{eq:op_policy} can be rearranged to express the reward function in terms of its corresponding optimal policy. The notation used for that optimal policy is
- Source: However, we can rearrange Eq.~\ref{eq:op_policy} to express the reward function in terms of its corresponding optimal policy $\pi_r$, the reference policy $\pi_\text{ref}$, and the unknown partition function $Z(\cdot)$.

### DPO [120]
- Target: `optimal model $\pi^*$.`
- Filter: It is unsuitable under criterion 1 because the probe already cues the answer as 'the corresponding model,' so the target 'optimal model π*' is largely just completing an obvious paraphrase rather than identifying a distinct concept.
- Probe: According to the paper "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", the reparameterization can be applied together with the ground-truth reward $r^*$ to the corresponding model, the
- Source: We can apply this reparameterization to the ground-truth reward $r^*$ and corresponding optimal model $\pi^*$.

### DPO [124]
- Target: `the optimal policy.`
- Filter: It is incorrectly formulated (criterion 5): the source says the preference probability is expressed in terms of the optimal policy rather than the reward model, and the target 'the optimal policy' is also too generic to support meaningful 5-way distractors.
- Probe: According to the paper "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", in the derivation of the DPO maximum likelihood objective for a parametrized policy $\pi_\theta$, the probability of human preference data is expressed in terms of the reward model, yielding
- Source: Now that we have the probability of human preference data in terms of the optimal policy rather than the reward model, we can formulate a maximum likelihood objective for a parametrized policy $\pi_\theta$.

### DPO [129]
- Target: `the preferred completions $y_w$`
- Filter: Unsuitable because the target is effectively already exposed in the probe text via the mention of preferred and dispreferred completions (y_w, y_l), making the completion tautological and violating criterion 2 (and effectively 1).
- Probe: According to the paper "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", and using the intuition given for the DPO loss $\mathcal{L}_\text{DPO}$ on preferred and dispreferred completions $(y_w, y_l)$, the gradient increases the likelihood of
- Source: Intuitively, the gradient of the loss function $\mathcal{L}_\text{DPO}$ increases the likelihood of the preferred completions $y_w$ and decreases the likelihood of dispreferred completions $y_l$.

### DPO [160]
- Target: `the equivalence class.`
- Filter: It fails criterion 5 because the probe/answer is incompletely formulated—the source fact says the normalized reward remains in the equivalence class of r(x, y), whereas the target given is only the fragment 'the equivalence class.'
- Probe: According to the paper "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", the fact that the added normalization term depends only on the prefix $x$ implies that $f(r; \pi_\text{ref}, \beta)(x, y)$ is a reward function in
- Source: Since the added normalization term is only a function of the prefix $x$, $f(r; \pi_\text{ref}, \beta)(x, y) $ is a reward function in the equivalence class of $r(x, y)$.

### DPO [170]
- Target: `$r_{\phi}$`
- Filter: It fails criterion 3 because the target is merely the paper’s variable notation "$r_{\phi}$", a trivially short symbol with little standalone conceptual depth for constructing a meaningful 5-option MCQA.
- Probe: In the paper "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", this same objective was optimized in prior works using the DPO-equivalent reward for the reward class of the function
- Source: This is the same objective optimized in prior works 
\citep{ziegler2020finetuning, stiennon2022learning, bai2022training, ouyang2022training} using the DPO-equivalent reward for the reward class of $r_{\phi}$.

### DPO [178]
- Target: `a reward function.`
- Filter: The probe is unsuitable because the target is a very generic short phrase ('a reward function') with little conceptual specificity, making it hard to build five meaningful distractors, and the probe is also awkwardly formulated relative to the source fact (criteria 3 and 5).
- Probe: In the paper "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", the authors contrast prior work that normalized rewards using a human completion baseline with the DPO reparameterization, which yields without requiring any baselines
- Source: In contrast the DPO reparameterization yields a reward function that does not require any baselines.

### DPO [185]
- Target: `$\mathcal{D}$.`
- Filter: It fails criterion 3 because the target is merely the notation symbol $\mathcal{D}$, a trivially short answer with little conceptual content, making it hard to create five meaningful MCQA options.
- Probe: According to the paper "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", the dataset of preferences used for all experiments is denoted by
- Source: For all experiments, algorithms learn a policy from a dataset of preferences $\mathcal{D}=\bigl\{x^{(i)}, y_w^{(i)}, y_l^{(i)}\bigr\}_{i=1}^N$.

### DPO [217]
- Target: `$y_w$.`
- Filter: It is unsuitable under criterion 3 because the target is just the paper’s notation "$y_w$", a trivially short symbol with little standalone conceptual content, making a 5-option MCQ unnatural and not very meaningful.
- Probe: In the paper "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", the Preferred-FT model is fine-tuned with supervised learning in the authors’ comparison of methods for training language models to adhere to human preferences, namely
- Source: In addition, we evaluate the \textbf{SFT} model as well as \textbf{Preferred-FT}, which is a model fine-tuned with supervised learning on the chosen completion $y_w$ from either the SFT model (in controlled sentiment and summarization) or a generic LM (in single-turn dialogue).

### DPO [223]
- Target: `two.`
- Filter: Unsuitable because the probe is incorrectly formulated and incomplete relative to the source fact, and the target is just the bare number "two," which is too trivial to support five meaningful multiple-choice options (criteria 3 and 5).
- Probe: According to the paper "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", the authors use
- Source: In our sentiment experiments, we use two implementations of PPO-GT, one of-the-shelf version \cite{leandro_von_werra_2023_7790115} as well as a modified version that normalizes rewards and further tunes hyperparameters to improve performance (we also use these modifications when running `normal' PPO with learned rewards).

### DPO [241]
- Target: `DPO's reward/KL tradeoff.`
- Filter: Unsuitable under criterion 1 because the target is essentially a tautological restatement of the probe itself: once the question names DPO and PPO and asks which tradeoff strictly dominates PPO, the answer is just 'DPO’s reward/KL tradeoff'.
- Probe: According to the paper "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", in the comparison of DPO and PPO on the reward versus average sequence-level KL frontier, the tradeoff said to strictly dominate PPO is
- Source: First, DPO and PPO optimize the same objective, but DPO is notably more efficient; DPO's reward/KL tradeoff strictly dominates PPO.

### DPO [247]
- Target: `reference completions.`
- Filter: It is incorrectly formulated (criterion 5): the probe says the win rate is computed 'against the evaluation set,' whereas the source fact states it is computed against reference completions in the test set, so the question-answer pair does not cleanly reflect the paper.
- Probe: According to the paper "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", when evaluating different methods on the test split of the TL;DR summarization dataset, the average win rate is computed against the evaluation set,
- Source: We evaluate different methods by sampling completions on the test split of TL;DR summarization dataset, and computing the average win rate against reference completions in the test set.

### DPO [305]
- Target: `a common misconception believed by 50\% of people`
- Filter: It fails criterion 1 because the target mostly just restates the probe itself ('a common misconception'), adding only a trivial qualifier ('believed by 50% of people'), so it would make a circular and weak MCQ.
- Probe: According to the paper "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", in the example involving a common misconception, we might want our language model to be aware of
- Source: Some of these goals and skillsets may not be desirable to imitate; for example, while we may want our AI coding assistant to \textit{understand} common programming mistakes in order to correct them, nevertheless, when generating code, we would like to bias our model toward the (potentially rare) high-quality coding ability present in its training data. Similarly, we might want our language model to be \textit{aware} of a common misconception believed by 50\% of people, but we certainly do not want the model to claim this misconception to be true in 50\% of queries about it! In other words, sel

### DPO [311]
- Target: `$Z(\cdot)$.`
- Filter: It is unsuitable under criterion 3 because the target is just the short symbolic expression "$Z(\cdot)$", which is too trivial and notation-specific to support five meaningful multiple-choice distractors.
- Probe: In the paper "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", when the authors rearrange Eq.~\ref{eq:op_policy} to express the reward function in terms of its corresponding optimal policy $\pi_r$ and the reference policy $\pi_\text{ref}$, the unknown partition function is
- Source: However, we can rearrange Eq.~\ref{eq:op_policy} to express the reward function in terms of its corresponding optimal policy $\pi_r$, the reference policy $\pi_\text{ref}$, and the unknown partition function $Z(\cdot)$.

### DPO [312]
- Target: `$\pi_\text{ref}(y \mid x)$`
- Filter: It fails criterion 1 and 3: the probe is essentially asking for the theorem’s notation that labels the reference model, so the answer is just a tautological symbol recall with little conceptual depth for constructing meaningful 5-way distractors.
- Probe: According to the paper "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", in the theorem on the DPO reparameterization, the given reference model is
- Source: \begin{theorem}\label{thm:main}
    Under mild assumptions, all reward classes consistent with the Plackett-Luce (and Bradley-Terry in particular) models can be represented with the reparameterization ${r(x, y) = \beta \log \frac{\pi(y\mid x)}{\pi_\text{ref}(y\mid x)}}$ for some model $\pi(y\mid x)$ and a given reference model $\pi_\text{ref}(y \mid x)$.
\end{theorem}

### DPO [313]
- Target: `probabilities are positive and sum to 1`
- Filter: It fails criterion 1 because the target is essentially a tautological restatement of the probe’s phrase 'is a valid distribution,' merely unpacking that this means the probabilities are positive and sum to 1.
- Probe: According to the paper "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", the equation specifying the reward function selected by the DPO reparameterization implies that $\pi(y\mid x)$ is a valid distribution because
- Source: We can alternatively view Theorem~\ref{thm:main} as specifying exactly which reward function within each equivalence class the DPO reparameterization selects, that is, the reward function satisfying:

\begin{equation}\label{eq:lag_p}
     \sum_{y}\underbrace{\pi_\text{ref}(y\mid x)\exp\left(\frac{1}{\beta}r(x, y)\right)}_{=\pi(y\mid x)\text{, using Thm.~\ref{thm:main} reparam.}} = 1,
\end{equation}

i.e., $\pi(y\mid x)$ is a valid distribution (probabilities are positive and sum to 1).

## FeatLLM candidates 17
### FeatLLM [5]
- Target: `this simple predictive model with the discovered features.`
- Filter: It fails criterion 5 because the target phrase is not self-contained and relies on the paper’s deictic wording ('this simple predictive model'), making the probe incorrectly formulated for a standalone MCQA.
- Probe: According to the paper "Large Language Models Can Automatically Engineer Features for Few-Shot Tabular Learning", at inference time, after the features have been discovered, the proposed `\textsf{FeatLLM}` framework uses
- Source: The proposed \textsf{FeatLLM} framework only uses this simple predictive model with the discovered features at inference time.

### FeatLLM [6]
- Target: `this simple predictive model with the discovered features.`
- Filter: It fails criterion 5 because the target phrase is incorrectly formulated for the probe: 'this simple predictive model' has no antecedent in the question text, so the question-answer pair does not stand alone properly even though the underlying fact could be made into a good MCQA.
- Probe: According to the paper "Large Language Models Can Automatically Engineer Features for Few-Shot Tabular Learning", at inference time, the proposed FeatLLM framework uses
- Source: The proposed \textsf{FeatLLM} framework only uses this simple predictive model with the discovered features at inference time.

### FeatLLM [31]
- Target: `at least one`
- Filter: Unsuitable under criterion 4 because the target is a minimal quantitative phrase ('at least one'), making it hard to create five natural, meaningful distractors rather than arbitrary numeric variants.
- Probe: According to the paper "Large Language Models Can Automatically Engineer Features for Few-Shot Tabular Learning", for end-to-end predictions in LLM-based tabular learning, the number of LLM inferences required per sample, making the method computationally expensive, is
- Source: For end-to-end predictions, at least one LLM inference per sample is required, making it computationally expensive.

### FeatLLM [42]
- Target: `rules.`
- Filter: It is incorrectly formulated (criterion 5): the probe awkwardly asks for the disease-identifying conditions 'as' something, whereas the source fact says the LLM generates rules that determine which feature conditions identify the disease.
- Probe: According to the paper "Large Language Models Can Automatically Engineer Features for Few-Shot Tabular Learning", in the proposed FeatLLM framework for feature engineering, for the task of predicting a particular disease, the LLM can directly infer and generate the conditions that identify the disease as
- Source: For instance, for the task of predicting a particular disease, the LLM can directly infer and generate rules that determine which feature conditions result in identifying the disease.

### FeatLLM [77]
- Target: `additional sources of information.`
- Filter: It is incorrectly formulated and awkwardly paraphrases the source clause, and the target 'additional sources of information' is too generic to support five meaningful distractors (criteria 3 and 5).
- Probe: According to the paper "Large Language Models Can Automatically Engineer Features for Few-Shot Tabular Learning", recent efforts have employed, alongside domain-specific prior knowledge and to impart appropriate inductive biases for model training,
- Source: Consequently, recent efforts have employed additional sources of information or introduced domain-specific prior knowledge to impart appropriate inductive biases for model training.

### FeatLLM [100]
- Target: `the input prompts of LLMs`
- Filter: It is unsuitable under criterion 3 because the target is a generic, low-depth phrase ('the input prompts of LLMs') that is too obvious from the wording 'feed the resulting text into' to support five meaningful distractors.
- Probe: In the paper "Large Language Models Can Automatically Engineer Features for Few-Shot Tabular Learning", after recent studies serialize tabular data into text, they feed the resulting text into
- Source: They typically propose serializing tabular data into text and feeding them into the input prompts of LLMs~\cite{dinh2022lift,hegselmann2023tabllm,wang2023anypredict}.

### FeatLLM [117]
- Target: `$\mathcal{D}$.`
- Filter: It fails criterion 3 because the target is just a bare notation symbol (\(\mathcal{D}\)) with little conceptual depth, making it difficult to create five meaningful answer choices beyond arbitrary variable names.
- Probe: According to the paper "Large Language Models Can Automatically Engineer Features for Few-Shot Tabular Learning", the notation used for the tabular dataset of $N$ labeled samples is
- Source: Let's consider a tabular dataset with $N$ labeled samples $\mathcal{D} = \{(\mathbf{x}^i, \mathbf{y}^i)\}_{i=1}^{N}$.

### FeatLLM [122]
- Target: `$k$ ($<N$) labeled samples.`
- Filter: It fails criterion 1 because the completion is essentially tautological with the probe’s own wording—given 'for the k-shot learning experiments,' answering 'k (<N) labeled samples' just restates the setup rather than testing a distinct concept.
- Probe: In the paper "Large Language Models Can Automatically Engineer Features for Few-Shot Tabular Learning", for the $k$-shot learning experiments, the model is trained by randomly sampling
- Source: For the $k$-shot learning experiments, only $k$ ($<N$) labeled samples are randomly sampled to train the model.

### FeatLLM [126]
- Target: `label information.`
- Filter: It is incorrectly formulated (criterion 5) because the source says the task description itself includes label information, not that 'label information' is a separate fourth component alongside the task, feature descriptions, and example demonstrations.
- Probe: According to the paper "Large Language Models Can Automatically Engineer Features for Few-Shot Tabular Learning", the basic information description includes the task to be solved, descriptions of features, example demonstrations, and
- Source: It includes a description of the task to be solved with label information, descriptions of features, and example demonstrations.

### FeatLLM [144]
- Target: `multiple rules.`
- Filter: It fails criterion 5 because the probe is incorrectly formulated relative to the source fact: the paper says the LLM can combine multiple rules using AND/OR, so the target 'multiple rules' makes the completion awkward and essentially circular rather than a distinct answerable concept.
- Probe: According to the paper "Large Language Models Can Automatically Engineer Features for Few-Shot Tabular Learning", when the prompt includes guidelines on the required response format for each answer class and the structure each rule should follow, the LLM can, without additional guidance, combine rules using logical operators like AND or OR into
- Source: Without additional guidance, the LLM can combine multiple rules using logical operators like AND or OR.

### FeatLLM [145]
- Target: `AND or OR.`
- Filter: Unsuitable under criterion 3 because the target is just a pair of very common logical operators ('AND or OR'), which is too trivial and shallow to support five meaningful multiple-choice distractors.
- Probe: In the paper "Large Language Models Can Automatically Engineer Features for Few-Shot Tabular Learning", when discussing the prompt guidance for the required response format and rule structure, the logical operators mentioned as examples of how the LLM can combine multiple rules without additional guidance are
- Source: Without additional guidance, the LLM can combine multiple rules using logical operators like AND or OR.

### FeatLLM [152]
- Target: `greater than.`
- Filter: It fails criterion 3 because the target is a trivially obvious short phrase ('greater than') directly implied by the symbol '>' in the probe, making it hard to create five meaningful distractors.
- Probe: According to the paper "Large Language Models Can Automatically Engineer Features for Few-Shot Tabular Learning", when the LLM alters expressions in generated rules during parsing, it might substitute the symbol "$>$" with
- Source: For instance, the LLM might alter expressions, like substituting ``$>$” or ``$<$” with phrases like ``greater than" or ``less than."

### FeatLLM [153]
- Target: `the LLM itself.`
- Filter: It is unsuitable under criterion 3 because the target, "the LLM itself," is a very short, generic phrase that is too obvious in the paper’s context and does not support five conceptually rich, distinguishable distractors.
- Probe: In the paper "Large Language Models Can Automatically Engineer Features for Few-Shot Tabular Learning", the authors address the challenges of parsing noisy text instead of building complex program code by leveraging
- Source: To address the challenges of parsing noisy text, instead of building complex program code, we leverage the LLM itself.

### FeatLLM [158]
- Target: `the LLM.`
- Filter: This probe is unsuitable under criterion 3 because the target, "the LLM," is a trivially short and overly generic phrase, making it hard to create five meaningful, discriminative answer options.
- Probe: According to the paper "Large Language Models Can Automatically Engineer Features for Few-Shot Tabular Learning", after including the function name, input and output descriptions, and inferred rules in the prompt for parsing noisy text, the prompt is then input into
- Source: To leverage these, we include the function name, input and output descriptions, and inferred rules in the prompt, then input them into the LLM.

### FeatLLM [166]
- Target: `$\mathbf{z}_k^i \in \{0, 1\}^R$.`
- Filter: It is unsuitable because the target is merely an exact notation/symbolic form from the paper rather than a substantive concept, making it hard to create five meaningful distractors beyond arbitrary symbol variants (criteria 3 and 4).
- Probe: In the paper "Large Language Models Can Automatically Engineer Features for Few-Shot Tabular Learning", the notation used for the generated binary feature from sample $\mathbf{x}^i$ for class $k$ is
- Source: Let us denote the generated binary feature from sample $\mathbf{x}^i$ for class $k$ as $\mathbf{z}_k^i \in \{0, 1\}^R$, where $R$ is the number of rules per each class and $c$ is the number of classes.

### FeatLLM [236]
- Target: `the second place`
- Filter: It is incorrectly formulated (criterion 5) because the source says FeatLLM 'consistently ranks as the top performer or secures the second place,' not that it consistently secures second place specifically.
- Probe: In the reported comparisons across datasets against other baseline models in the paper "Large Language Models Can Automatically Engineer Features for Few-Shot Tabular Learning", \textsf{FeatLLM} consistently secures
- Source: \textsf{FeatLLM} consistently ranks as the top performer or secures the second place when compared with other baseline models.

### FeatLLM [301]
- Target: ```$>$” or ``$<$”`
- Filter: It fails criterion 3 because the target is just a pair of generic comparison symbols (">" and "<") with little conceptual depth, making it difficult to form a strong 5-option MCQA beyond arbitrary symbol variants.
- Probe: In the paper "Large Language Models Can Automatically Engineer Features for Few-Shot Tabular Learning", when the LLM introduces noisy syntactic changes during rule generation, the comparison symbols it might replace are
- Source: For instance, the LLM might alter expressions, like substituting ``$>$” or ``$<$” with phrases like ``greater than" or ``less than."

## GRPO candidates 23
### GRPO [2]
- Target: `DeepSeek-Coder-Base-v1.5 7B`
- Filter: It fails criterion 5 because the probe is incorrectly formulated: the paper says DeepSeekMath~7B continues pre-training from the base model DeepSeek-Coder-Base-v1.5 7B, not that the natural language and code data are 'from' that model.
- Probe: According to the paper "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", DeepSeekMath~7B continues pre-training from the base model using 120B math-related tokens sourced from Common Crawl, together with natural language and code data from
- Source: In this paper, we introduce DeepSeekMath~7B, which continues pre-training DeepSeek-Coder-Base-v1.5 7B  with 120B math-related tokens sourced from Common Crawl, together with natural language and code data.

### GRPO [28]
- Target: `mathematical instruction tuning`
- Filter: The probe is incorrectly formulated (criterion 5) because it implies the chain-of-thought, program-of-thought, and tool-integrated reasoning data were used during pre-training, whereas the source states those data were used for the subsequent mathematical instruction tuning.
- Probe: In the paper "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", after pre-training DeepSeekMath-Base with chain-of-thought, program-of-thought, and tool-integrated reasoning data, the authors apply to DeepSeekMath-Base
- Source: After pre-training, we apply mathematical instruction tuning to DeepSeekMath-Base with chain-of-thought \citep{cot}, program-of-thought \citep{pot,pal}, and tool-integrated reasoning \citep{tora} data.

### GRPO [35]
- Target: `training resources.`
- Filter: It is incorrectly formulated (criterion 5): the source fact says GRPO 'significantly reduc[es] training resources,' whereas the probe rephrases this as 'preserving ___,' and the target 'training resources' is also too generic to support strong, meaningful distractors.
- Probe: According to the paper "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", by foregoing the critic model and estimating the baseline from group scores, the Group Relative Policy Optimization (GRPO) algorithm significantly reduces computational overhead, preserving
- Source: GRPO foregoes the critic model, instead estimating the baseline from group scores, significantly reducing training resources.

### GRPO [54]
- Target: `all mathematical benchmarks.`
- Filter: Unsuitable because the target is essentially a tautological paraphrase of the probe text itself ('the benchmarks adopted in this paper' ≈ 'all mathematical benchmarks') and the cloze is also awkwardly formulated relative to the source fact, making it fail criteria 1 and 5.
- Probe: According to the paper "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", the authors state that although training on arXiv papers is common in many math-related papers, it brings no notable improvements on the benchmarks adopted in this paper,
- Source: Although training on arXiv papers is common, especially in many math-related papers, it brings no notable improvements on all mathematical benchmarks adopted in this paper.

### GRPO [78]
- Target: `math pre-training.`
- Filter: It fails criterion 5 because the probe is incorrectly formulated relative to the source fact: the paper says DeepSeekMath-Base surpasses open-source base models regardless of whether those other models had undergone math pre-training, not that it surpasses them 'with math pre-training.'
- Probe: According to the paper "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", on the English benchmarks discussed for DeepSeekMath-Base, DeepSeekMath-Base still surpasses the open-source base models with
- Source: On English benchmarks, DeepSeekMath-Base is competitive with the closed-source Minerva 540B \citep{minerva}, and surpasses all open-source base models (e.g., Mistral 7B \citep{mistral} and Llemma-34B \citep{llemma}), regardless of whether they've undergone math pre-training or not, often by a significant margin.

### GRPO [79]
- Target: `Chinese benchmarks.`
- Filter: The probe is incorrectly formulated (criterion 5): the source says the model is superior on Chinese benchmarks because the authors include high-quality non-English pre-training data, not specifically 'data on Chinese benchmarks,' so the question-answer pairing distorts the original fact.
- Probe: According to the paper "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", DeepSeekMath-Base is described as being superior on the following benchmarks, likely because the authors do not follow previous works to collect English-only math pre-training data and also include high-quality non-English data on
- Source: Notably, DeepSeekMath-Base is superior on Chinese benchmarks, likely because we don't follow previous works \citep{minerva,llemma} to collect English-only math pre-training data, and also include high-quality non-English ones.

### GRPO [88]
- Target: `Massive Multitask Language Understanding (MMLU).`
- Filter: It is incorrectly formulated (criterion 5) because the source fact names multiple benchmarks used for assessing natural language understanding, reasoning, and code (MMLU, BBH, HumanEval, and MBPP), so asking for a single benchmark makes the target non-unique.
- Probe: According to the paper "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", the benchmark the authors use to evaluate DeepSeekMath-Base as part of their assessment of natural language understanding, reasoning, and code is
- Source: \item \textbf{Natural Language Understanding, Reasoning, and Code}: To build a comprehensive profile of models' general understanding, reasoning, and coding capabilities, we evaluate DeepSeekMath-Base on the Massive Multitask Language Understanding (MMLU) benchmark \citep{mmlu} which encompasses 57 multiple-choice tasks covering diverse subjects, BIG-Bench Hard (BBH) \citep{bbh} which consists of 23 challenging tasks that mostly require multi-step reasoning to solve, as well as HumanEval \citep{codex} and MBPP \citep{mbpp} which are widely used to evaluate code language models.

### GRPO [105]
- Target: `160B tokens.`
- Filter: It fails criterion 1 because the probe itself gives an obvious numerical sequence ('40B, 80B, 120B ...') that makes '160B tokens' a trivial continuation rather than a meaningful knowledge-based MCQA target.
- Probe: In the paper "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", the authors assess the volume of data preserved through pre-training experiments on the top 40B, 80B, and 120B tokens, as well as
- Source: The volume of data preserved is assessed through pre-training experiments on the top 40B, 80B, 120B, and 160B tokens.

### GRPO [174]
- Target: `the two coding benchmarks.`
- Filter: Unsuitable because the target is a vague generic phrase rather than the specific benchmarks (e.g., HumanEval and MBPP), making the probe poorly formulated and not conducive to 5 meaningful distractors (criteria 3 and 5).
- Probe: According to the paper "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", by including code tokens for continual training, DeepSeekMath-Base 7B effectively maintains the performance of DeepSeek-Coder-Base-v1.5 on
- Source: Additionally, by including code tokens for continual training, DeepSeekMath-Base 7B effectively maintains the performance of DeepSeek-Coder-Base-v1.5 on the two coding benchmarks.

### GRPO [182]
- Target: `Chinese K-12 mathematical problems.`
- Filter: It fails criterion 1 because the target, "Chinese K-12 mathematical problems," is essentially a restatement of the probe's own description of a Chinese mathematical dataset spanning 76 sub-topics with CoT and tool-integrated solutions.
- Probe: According to the paper "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", the authors collect a Chinese mathematical dataset spanning 76 sub-topics with solutions annotated in both CoT and tool-integrated reasoning format, including
- Source: We collect Chinese K-12 mathematical problems spanning 76 sub-topics such as linear equations, with solutions annotated in both CoT and tool-integrated reasoning format.

### GRPO [225]
- Target: `$\mathcal{J}_{GRPO}(\theta)$`
- Filter: Unsuitable under criterion 1 because the probe is effectively tautological: it asks for the name/symbol of 'the objective' being maximized, and the target is merely the notation for that objective rather than a distinct concept.
- Probe: In the paper "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", when Group Relative Policy Optimization (GRPO) samples a group of outputs $\{o_1, o_2, \cdots, o_G\}$ from the old policy $\pi_{\theta_{old}}$ for each question $q$, it maximizes the objective
- Source: More specifically, for each question $q$, GRPO samples a group of outputs $\{o_1, o_2, \cdots, o_G\}$  from the old policy  $\pi_{\theta_{old}}$  and then optimizes the policy model by maximizing the following objective:

\begin{equation}
\footnotesize
\begin{split}
    \mathcal{J}_{GRPO}(\theta) &= \mathbb{E}{[q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)]}  \\
    & \frac{1}{G}\sum_{i=1}^G\frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left\{ \min \left[ \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} | q, o_{i,<t})} \hat{A}_{i,t}, \text{clip} \left( \frac{\pi_\theta(o_{i

### GRPO [234]
- Target: `$G$ rewards.`
- Filter: It is unsuitable because the target is a trivially short symbolic quantity ("G rewards") with little conceptual depth, making it hard to construct five meaningful distractors beyond arbitrary numeric variants (criteria 3 and 4).
- Probe: According to the paper "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", in the Outcome Supervision RL with GRPO setting, when a reward model scores a group of outputs sampled from the old policy model $\pi_{\theta_{old}}$ for each question $q$, the number of rewards obtained correspondingly is
- Source: A reward model is then used to score the outputs, yielding $G$ rewards  $\mathbf{r}=\{r_1, r_2, \cdots, r_G\}$ correspondingly.

### GRPO [238]
- Target: `$\mathcal{J}_{GRPO}(\theta)$`
- Filter: It fails criterion 1 because the probe already specifies the GRPO setting and asks for the objective maximized by the policy, so answering with the notation $\mathcal{J}_{GRPO}(\theta)$ is essentially just restating 'the GRPO objective' in symbolic form rather than testing a distinct concept.
- Probe: According to the paper "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", in the outcome supervision setting of GRPO, after the normalized reward $\widetilde{r}_i$ is assigned as the advantage $\hat{A}_{i,t}$ for all tokens in an output $o_i$, the objective maximized by the policy is
- Source: Outcome supervision provides the normalized reward at the end of each output $o_i$  and sets the advantages  $\hat{A}_{i, t}$  of all tokens in the output as the normalized reward, i.e., $\hat{A}_{i, t} = \widetilde{r}_i = \frac{r_i- {\rm mean}(\mathbf{r})}{{\rm std}(\mathbf{r})}$, and then optimizes the policy by maximizing the objective defined in equation (\ref{eq:GRPO-obj}).

### GRPO [242]
- Target: `reasoning step.`
- Filter: It fails criterion 5 because the probe is incorrectly formulated: the source fact says process supervision gives a reward at the end of each reasoning step, but the probe's phrasing about completing a derivation does not accurately or naturally reflect that fact.
- Probe: According to the paper "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", in the discussion of process supervision with GRPO for complex mathematical tasks, process supervision provides a reward after the model completes its derivation in the
- Source: Following \cite{wang2023math}, we also explore process supervision, which provides a reward at the end of each reasoning step.

### GRPO [257]
- Target: `to investigate the impact of RL.`
- Filter: It is incorrectly formulated (criterion 5) because the provided target is an overly generic purpose statement—whereas the source fact specifically says the exclusion was to study RL's impact on benchmarks lacking data during the RL phase—so the question-answer pair does not precisely reflect the source.
- Probe: In the paper "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", the authors exclude the other SFT questions from the RL training data when conducting RL based on DeepSeekMath-Instruct 7B on GSM8K and MATH chain-of-thought-format questions in order
- Source: We exclude other SFT questions to investigate the impact of RL on benchmarks that lack data throughout the RL phase.

### GRPO [258]
- Target: `the RL phase.`
- Filter: It is unsuitable under criterion 3 because the target, "the RL phase," is a short generic phrase with little conceptual depth, making it hard to create five meaningful, discriminative answer options.
- Probe: In the paper "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", the authors exclude other SFT questions to investigate the impact of reinforcement learning on benchmarks that lack data throughout
- Source: We exclude other SFT questions to investigate the impact of RL on benchmarks that lack data throughout the RL phase.

### GRPO [301]
- Target: `Data Source $\mathcal{D}$.`
- Filter: Unsuitable under criterion 1 because the probe is essentially definitional/circular: 'the component that determines the training data' directly paraphrases the answer 'Data Source \(\mathcal{D}\)'.
- Probe: In the paper "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", in the unified paradigm, the component that determines the training data is
- Source: There exist three key components: 1) \textit{Data Source $\mathcal{D}$}, which determines the training data; 2) \textit{Reward Function $\pi_{{rf}}$}, which is the source of the training reward signal; 3) \textit{Algorithm $\mathcal{A}$}: which processes the training data and the reward signal to the gradient coefficient $GC$ that determines the magnitude of the penalty or reinforcement for the data.

### GRPO [306]
- Target: `filtered outputs.`
- Filter: It fails criterion 5 because the probe is awkwardly and imprecisely formulated—RFT trains on filtered sampled outputs, so asking 'sampled outputs ..., namely' makes the target a generic phrase rather than a clearly distinguished concept for a 5-option MCQA.
- Probe: According to the paper "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", Rejection Sampling Fine-tuning (RFT) further fine-tunes the SFT model on sampled outputs from the SFT model, namely
- Source: \item \textbf{Rejection Sampling Fine-tuning (RFT)}: RFT further fine-tunes the SFT model on the filtered outputs sampled from the SFT model based on SFT questions. RFT filters the outputs based on the correctness of their answers.

### GRPO [330]
- Target: `iterative RL.`
- Filter: Unsuitable under criterion 1 because the probe is essentially tautological: describing the method as the RL variant with 'two rounds of iteration' nearly restates the answer 'iterative RL.'
- Probe: According to the paper "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", the reinforcement learning variant the authors explore in their experiments, where they conduct two rounds of iteration, is
- Source: Furthermore, we explore the iterative RL, in our experiments, we conduct two rounds of iteration.

### GRPO [336]
- Target: `the correct response from TopK.`
- Filter: It fails criterion 5 because the target is incorrectly/incompletely formulated relative to the source fact: the paper attributes the improvement to 'boosting the correct response from TopK,' not merely 'the correct response from TopK.'
- Probe: According to the paper "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", after evaluating the Pass@K and Maj@K accuracy of the Instruct and RL models on two benchmarks, the improvement appears to be attributed to
- Source: These findings indicate that RL enhances the model's overall performance by rendering the output distribution more robust, in other words, \textbf{it seems that the improvement is attributed to boosting the correct response from TopK rather than the enhancement of fundamental capabilities.}

### GRPO [339]
- Target: `three.`
- Filter: It fails criterion 3 because the target is just the bare number "three," a trivially short answer with little conceptual depth, making a 5-option MCQ largely arbitrary among numeric distractors.
- Probe: According to the paper "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", Equation \ref{eq:objective} summarizes the number of key components for the unified paradigm discussed in the section on achieving more effective RL as
- Source: As summarized in Equation \ref{eq:objective}, there exist three key components: Data Source, Algorithm, and Reward Function.

### GRPO [348]
- Target: `the model parameter.`
- Filter: It is incorrectly formulated (criterion 5): the probe is an ungrammatical fragment that does not clearly or accurately express the source fact, and the target phrase "the model parameter" is too generic to support natural 5-option discrimination.
- Probe: In the paper "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", in the RL pipeline algorithms discussion, processing the data and reward signal to the gradient coefficient updates
- Source: Algorithms process the data and reward signal to the gradient coefficient to update the model parameter.

### GRPO [393]
- Target: `$\mathcal{D}$`
- Filter: It fails criterion 3 because the target is only the bare symbol $\mathcal{D}$ from an equation, which is too notation-specific and lacks enough conceptual content to support five meaningful multiple-choice options.
- Probe: According to the paper "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", in the unified paradigm used to analyze training methods such as SFT, RFT, DPO, PPO, and GRPO, the pairs $(q,o)$ are sampled in the expectation defining the gradient with respect to $\theta$ from
- Source: Generally, the gradient with respect to the parameter $\theta$ of a training method can be written as:

\begin{equation}
    \nabla_{\theta}\mathcal{J}_{\textcolor{red}{\mathcal{A}}}(\theta) = \mathbb{E}[\underbrace{(q,o) \sim \textcolor{red}{\mathcal{D}}}_{Data \ Source}]\left( \frac{1}{|o|} \sum_{t=1}^{|o|}  \underbrace{GC_{{\mathcal{A}}}(q, o, t, \textcolor{red}{\pi_{{rf}}})}_{Gradient \ Coefficient}  \nabla_{\theta}\log \pi_{\theta}(o_t | q, o_{<t})\right).
\label{eq:objective}
\end{equation}

## GSPO candidates 5
### GSPO [29]
- Target: `$\pi_\theta (y | x)=\prod_{t=1}^{|y|} \pi_\theta (y_t | x, y_{<t} )$`
- Filter: It fails criterion 1 (tautological/circular) because the probe already specifies that the likelihood is written using the token-level autoregressive factorization over the tokens in y, so the target is just that factorization spelled out.
- Probe: According to the paper "Group Sequence Policy Optimization", the likelihood of a response $y$ to a query $x$ under the policy $\pi_\theta$ is denoted, using the token-level autoregressive factorization over the tokens in $y$, by
- Source: Given a response $y$ to a query $x$, its likelihood under the policy $\pi_\theta$ is denoted as $\pi_\theta (y | x)=\prod_{t=1}^{|y|} \pi_\theta (y_t | x, y_{<t} )$ where $|y|$ denotes the number of tokens in $y$.

### GSPO [77]
- Target: `GSPO.`
- Filter: Unsuitable by criterion 1: the answer is tautologically recoverable from the probe itself because 'GSPO' is just the acronym of the algorithm name 'Group Sequence Policy Optimization' already written in the question.
- Probe: According to the paper "Group Sequence Policy Optimization", the abbreviation the authors use for the proposed algorithm is
- Source: Based on this straightforward observation, we propose the \textbf{Group Sequence Policy Optimization (GSPO)} algorithm.

### GSPO [83]
- Target: `dramatic fluctuations.`
- Filter: It fails criterion 5 because the probe-answer pair is incompletely formulated relative to the source fact—the source says a few token likelihood changes cause "dramatic fluctuations of the sequence-level importance ratio," whereas the target alone is just the generic phrase "dramatic fluctuations."
- Probe: In the paper "Group Sequence Policy Optimization", the discussion of length normalization in $s_{i}(\theta)$ and the sequence-level importance ratio states that likelihood changes of a few tokens can cause
- Source: Otherwise, the likelihood changes of a few tokens can result in dramatic fluctuations of the sequence-level importance ratio, and the importance ratios of responses with different lengths will require varying clipping ranges.

### GSPO [125]
- Target: `learning signal.`
- Filter: Unsuitable because the target is a trivially generic short phrase and the probe is incompletely formulated relative to the source, which specifically says GSPO provides 'a more reliable and effective learning signal' (criteria 3 and 5).
- Probe: According to the paper "Group Sequence Policy Optimization", GSPO's sequence-level approach provides, in contrast to GRPO's token-level gradient estimates, a
- Source: In contrast, GSPO's sequence-level approach provides a more reliable and effective learning signal.

### GSPO [151]
- Target: `partial rollout`
- Filter: The probe is incorrectly formulated relative to the source fact (criterion 5), because the paper lists partial rollout, multi-turn RL, and training-inference disaggregated frameworks as separate beneficial scenarios, whereas the probe syntactically merges them into 'frameworks with partial rollout.'
- Probe: In the paper "Group Sequence Policy Optimization", the authors state that directly using the likelihoods returned by the inference engine for optimization, instead of recomputing them with the training engine, is especially beneficial for multi-turn RL and training-inference disaggregated frameworks with
- Source: This can be especially beneficial in scenarios like partial rollout and multi-turn RL and in the training-inference disaggregated frameworks.

## OFT candidates 18
### OFT [25]
- Target: `the lack of a measure.`
- Filter: It is incorrectly formulated and too underspecified: the source fact says the key difficulty is the lack of a measure for quantifying preservation of pretrained generative ability, whereas the target 'the lack of a measure' is too generic to support meaningful 5-way distractors (criteria 3 and 5).
- Probe: In the paper "Controlling Text-to-Image Diffusion by Orthogonal Finetuning", the key difficulty the authors identify in designing a finetuning strategy for preserving pretrained generative ability is
- Source: A key difficulty is the lack of a measure for quantifying the preservation of pretrained generative ability.

### OFT [34]
- Target: `a minimal difference.`
- Filter: It fails criterion 5 because the probe is incorrectly formulated: the source fact says a good finetuned model should have a minimal difference in hyperspherical energy compared to the pretrained model, whereas the target 'a minimal difference' is incomplete and too underspecified to support meaningful distractors.
- Probe: According to the paper "Controlling Text-to-Image Diffusion by Orthogonal Finetuning", a good finetuned model should have
- Source: We hypothesize that a good finetuned model should have a minimal difference in hyperspherical energy compared to the pretrained model.

### OFT [61]
- Target: `generalizable neural networks.`
- Filter: Unsuitable because it is largely tautological and awkwardly formulated: the probe already specifies 'neural networks trained using orthogonal transformation for classification problems,' and the target merely restates that they are 'generalizable neural networks' rather than eliciting a distinct concept (criteria 1 and 5).
- Probe: According to the paper "Controlling Text-to-Image Diffusion by Orthogonal Finetuning", neural networks trained using orthogonal transformation for classification problems are
- Source: \cite{liu2021orthogonal} shows that orthogonal transformation is sufficiently flexible to train generalizable neural networks for classification problems.

### OFT [68]
- Target: `text-to-image diffusion models`
- Filter: Unsuitable because the target is effectively exposed in the probe itself via the paper title “Controlling Text-to-Image Diffusion by Orthogonal Finetuning,” making the answer “text-to-image diffusion models” nearly verbatim leakage (criterion 2).
- Probe: According to the paper "Controlling Text-to-Image Diffusion by Orthogonal Finetuning", the class of generative models that Orthogonal Finetuning is proposed to guide toward better controllability is
- Source: \item We propose a novel finetuning method -- Orthogonal Finetuning for guiding text-to-image diffusion models towards better controllability.

### OFT [91]
- Target: `generative adversarial networks.`
- Filter: It fails criterion 5 because the source fact states that both generative adversarial networks and autoregressive models have been studied, so the probe is incorrectly formulated as if only one answer were correct.
- Probe: According to the paper "Controlling Text-to-Image Diffusion by Orthogonal Finetuning", other than diffusion models, the kind of networks that have also been studied in text-to-image generation is
- Source: Other than diffusion models, generative adversarial networks~\cite{reed2016generative,zhang2017stackgan,xu2018attngan,li2019controllable} and autoregressive models~\cite{ramesh2021zero,ding2021cogview,wu2022nuwa,yuscaling} have also been studied in text-to-image generation.

### OFT [103]
- Target: `a reconstruction loss.`
- Filter: It fails criterion 5 because the source fact says DreamBooth uses two losses—a reconstruction loss and a class-specific prior preservation loss—so the probe is incorrectly formulated as if a single completion were sufficient.
- Probe: According to the paper "Controlling Text-to-Image Diffusion by Orthogonal Finetuning", when DreamBooth finetunes the text-to-image diffusion model with a customized token and a few subject images, it uses an objective consisting of
- Source: With a customized token and a few subject images, DreamBooth~\cite{ruiz2023dreambooth} finetunes the text-to-image diffusion model using a reconstruction loss and a class-specific prior preservation loss.

### OFT [105]
- Target: `the DreamBooth framework.`
- Filter: It fails criterion 5 (incorrectly formulated): the paper says OFT adopts the DreamBooth framework, but the thing used instead of naive small-learning-rate finetuning is orthogonal transformations, not DreamBooth.
- Probe: According to the paper "Controlling Text-to-Image Diffusion by Orthogonal Finetuning", instead of naive finetuning with a small learning rate, OFT adopts
- Source: OFT adopts the DreamBooth framework, but instead of performing naive finetuning with a small learning rate, OFT finetunes the model with orthogonal transformations.

### OFT [130]
- Target: `$z$.`
- Filter: It fails criterion 3 because the target is just the single-variable symbol "$z$", which is too trivially short and lacks enough conceptual content to support five meaningful multiple-choice options.
- Probe: According to the paper "Controlling Text-to-Image Diffusion by Orthogonal Finetuning", in the toy experiment with a standard convolutional autoencoder trained on flower images, the element output of the convolution kernel $\bm{w}$ under the standard inner product is denoted by
- Source: In the training stage, we use the standard inner product to produce the feature map ($z$ denotes the element output of the convolution kernel $\bm{w}$ and $\bm{x}$ is the input in the sliding window).

### OFT [166]
- Target: `\bm{x}\in\mathbb{R}^d`
- Filter: It fails criterion 3 because the target is merely an arbitrary mathematical notation snippet for a variable (the input vector), with little conceptual depth and weak natural distractors for a 5-option MCQ.
- Probe: According to the paper "Controlling Text-to-Image Diffusion by Orthogonal Finetuning", in the fully connected layer example used to explain OFT, the notation for the input vector is
- Source: The output vector $\thickmuskip=2mu \medmuskip=2mu \bm{z}\in\mathbb{R}^n$ of this fully connected layer is computed by $\thickmuskip=2mu \medmuskip=2mu \bm{z}=\bm{W}^\top\bm{x}$ where $\thickmuskip=2mu \medmuskip=2mu \bm{x}\in\mathbb{R}^d$ is the input vector.

### OFT [188]
- Target: `1.`
- Filter: It is unsuitable under criterion 4 because the target is a bare numeric value ('1') for a determinant property, making it hard to construct five natural, meaningful distractors beyond a few obvious alternatives like -1 or ±1.
- Probe: According to the paper "Controlling Text-to-Image Diffusion by Orthogonal Finetuning", the orthogonal matrices generated by the Cayley parameterization can have determinant
- Source: Such an efficiency comes at a small price -- the Cayley parameterization can only produce orthogonal matrices with determinant $1$ which belongs to the special orthogonal group.

### OFT [198]
- Target: `$\bm{R}_i=\bm{R}_j,\forall i\neq j$`
- Filter: Unsuitable under criterion 1 because the probe is essentially tautological: asking what condition holds when the block matrices are 'shared' is answered directly by stating that all block matrices are equal, which just restates the premise.
- Probe: According to the paper "Controlling Text-to-Image Diffusion by Orthogonal Finetuning", the condition imposed on the block matrices when they are optionally shared in the $r$-block diagonal orthogonal matrix parameterization to further reduce the number of parameters is
- Source: We can optionally share the block matrix to further reduce the number of parameters, \emph{i.e.}, $\thickmuskip=2mu \medmuskip=2mu\bm{R}_i=\bm{R}_j,\forall i\neq j$.

### OFT [206]
- Target: `the flexibility of original OFT.`
- Filter: It fails criterion 1 because the probe is essentially a sentence fragment whose completion directly restates the source claim, making the answer a tautological phrase completion rather than a concept that supports meaningful 5-way discrimination.
- Probe: In the paper "Controlling Text-to-Image Diffusion by Orthogonal Finetuning", constraining the finetuned model to be within a small neighborhood of the pretrained model further limits
- Source: We can further limit the flexibility of original OFT by constraining the finetuned model to be within a small neighborhood of the pretrained model.

### OFT [222]
- Target: `smaller $\epsilon$`
- Filter: Unsuitable under criterion 4 because the target is just the directional comparative 'smaller ε,' making it hard to construct five natural, meaningful distractor options beyond variants like larger/smaller/intermediate values.
- Probe: According to the paper "Controlling Text-to-Image Diffusion by Orthogonal Finetuning", in the COFT finetuning experiment where $\epsilon$ controls the flexibility of finetuning, the choice of $\epsilon$ that makes the COFT-finetuned model behave increasingly similar to the pretrained text-to-image diffusion model is
- Source: With smaller $\epsilon$, the COFT-finetuned model behaves increasingly similar to the pretrained text-to-image diffusion model.

### OFT [227]
- Target: `$\bm{R}$.`
- Filter: It is unsuitable under criterion 3 because the target is just the single symbol $\bm{R}$, a trivially short equation variable with little standalone conceptual depth, making natural 5-option distractors weak and overly notation-dependent.
- Probe: According to the paper "Controlling Text-to-Image Diffusion by Orthogonal Finetuning", in OFT’s original forward pass in Eq. \eqref{eq:oft_general}, the learnable matrix is
- Source: In contrast to OFT's original forward pass in Eq.~\eqref{eq:oft_general} where only $\bm{R}$ is learnable, we have both the diagonal matrix $\bm{D}$ and the orthogonal matrix $\bm{R}$ learnable.

### OFT [255]
- Target: `a stronger regularization.`
- Filter: It fails criterion 5 because the probe/answer is imprecisely formulated: the source fact states the block-diagonal parameterization is 'a stronger regularization of the orthogonal matrix,' and the truncated target 'a stronger regularization' is too incomplete and generic.
- Probe: According to the paper "Controlling Text-to-Image Diffusion by Orthogonal Finetuning", the block-diagonal parameterization in the OFT framework can be viewed as
- Source: The block-diagonal parameterization can also be viewed as a stronger regularization of the orthogonal matrix.

### OFT [259]
- Target: `\bm{W}=\bm{R}\bm{W}^0`
- Filter: It is unsuitable under criterion 1 because the probe already states that the equivalent weight matrix is obtained by multiplying the learned orthogonal matrix R into the pretrained weight matrix W^0, so the target W=RW^0 is essentially just a direct restatement of the question.
- Probe: According to the paper "Controlling Text-to-Image Diffusion by Orthogonal Finetuning", in the inference stage of OFT, the authors obtain the equivalent weight matrix by multiplying the learned orthogonal matrix $\bm{R}$ into the pretrained weight matrix $\bm{W}^0$, which is
- Source: In the inference stage, we can simply multiply the learned orthogonal matrix $\bm{R}$ into the pretrained weight matrix $\bm{W}^0$ and obtain an equivalent weight matrix $\thickmuskip=2mu \medmuskip=2mu \bm{W}=\bm{R}\bm{W}^0$.

### OFT [265]
- Target: `the same layer.`
- Filter: It is incorrectly formulated under criterion 5 because the target 'the same layer' omits the essential qualifier from the source fact—'where LoRA is used'—so the question-answer pair is incomplete and not precise enough for a good MCQA.
- Probe: According to the paper "Controlling Text-to-Image Diffusion by Orthogonal Finetuning", to ensure a fair comparison to LoRA in the general settings experiment, OFT or COFT are only applied to
- Source: To ensure a fair comparison to LoRA, we only apply OFT or COFT to the same layer where LoRA is used.

### OFT [281]
- Target: `proper $\epsilon$.`
- Filter: It fails criterion 1 because the probe is essentially tautological—the answer 'proper ε' just restates the vague description in the question rather than identifying a distinct concept that could support meaningful distractors.
- Probe: According to the paper "Controlling Text-to-Image Diffusion by Orthogonal Finetuning", for COFT in subject-driven generation, the kind of $\epsilon$ that makes both the learning rate and the iteration number effortless to set is
- Source: For COFT with a proper $\epsilon$, both the learning rate and the iteration number become effortless to set.

## QLoRA candidates 16
### QLoRA [15]
- Target: `chatbots.`
- Filter: It is incorrectly formulated (criterion 5) because the source says the benchmarks are not trustworthy for accurately evaluating the performance levels of chatbots, not simply for assessing 'chatbots,' and the target is also too generic to support strong distractors.
- Probe: According to the paper "QLoRA: Efficient Finetuning of Quantized LLMs", the authors find that current benchmark evaluations are not trustworthy for accurately assessing
- Source: Furthermore, we find that current chatbot benchmarks are not trustworthy to accurately evaluate the performance levels of chatbots.

### QLoRA [16]
- Target: `Finetuning large language models (LLMs).`
- Filter: It fails criterion 1 because the target simply restates the probe’s wording ('a highly effective way to improve the performance of LLMs and add/remove behaviors is...'), making the answer essentially tautological.
- Probe: According to the paper "QLoRA: Efficient Finetuning of Quantized LLMs", a highly effective way to improve the performance of large language models (LLMs) and to add desirable or remove undesirable behaviors is
- Source: Finetuning large language models (LLMs) is a highly effective way to improve their performance, \citep{min2021metaicl, wei2021finetuned, ouyang2022training, wang2022super, wang2022self, liu2022few} and to add desirable or remove undesirable behaviors \citep{ouyang2022training,askell2021general,bai2022training}.

### QLoRA [18]
- Target: `780 GB.`
- Filter: It fails criterion 5 because the probe asks for an exact memory requirement, but the source fact states it requires more than 780 GB, so the provided target '780 GB' is not precisely formulated.
- Probe: According to the paper "QLoRA: Efficient Finetuning of Quantized LLMs", regular 16-bit finetuning of a LLaMA 65B parameter model requires
- Source: However, finetuning very large models is prohibitively expensive; regular 16-bit finetuning of a LLaMA 65B parameter model~\citep{touvron2023llama} requires more than 780 GB of GPU memory.

### QLoRA [48]
- Target: `a cheap alternative.`
- Filter: It is incorrectly formulated (criterion 5) because the source fact says model-based evaluation provides a cheap alternative to human annotation, while the probe truncates that relation and leaves an overly vague target phrase.
- Probe: According to the paper "QLoRA: Efficient Finetuning of Quantized LLMs", in the authors' discussion of chatbot benchmark evaluation, model-based evaluation provides
- Source: As such, we highlight that model-based evaluation while providing a cheap alternative to human-annotation also has its uncertainties.

### QLoRA [65]
- Target: `\mathbf{X}^{\text{FP32}}`
- Filter: This fails criterion 1 because the probe is essentially tautological: it asks which FP32 tensor is recovered, and the target is just the corresponding notation \(\mathbf{X}^{\text{FP32}}\) already implied by the question.
- Probe: According to the paper "QLoRA: Efficient Finetuning of Quantized LLMs", in the description of quantizing an FP32 tensor into an Int8 tensor, the dequantization function $\text{dequant}(c^{\text{FP32}}, \mathbf{X}^{\text{Int8}})$ recovers the FP32 tensor
- Source: Dequantization is the inverse:

\begin{equation}
    \text{dequant}(c^{\text{FP32}}, \mathbf{X}^{\text{Int8}}) = \frac{\mathbf{X}^{\text{Int8}}}{c^{\text{FP32}}} = \mathbf{X}^{\text{FP32}}
\end{equation}

### QLoRA [72]
- Target: `$B$.`
- Filter: It fails criterion 3 because the target is merely the single symbol "$B$", a trivially short variable name with little conceptual content and no natural set of meaningful distractors.
- Probe: According to the paper "QLoRA: Efficient Finetuning of Quantized LLMs", when the input tensor $\mathbf{X}\in \mathbb{R}^{b\times h}$ is flattened and sliced into contiguous blocks for independent quantization, the block size is
- Source: This can be formalized as follows: We chunk the input tensor  $\mathbf{X}\in \mathbb{R}^{b\times h}$ into $n$ contiguous blocks of size $B$ by flattening the input tensor and slicing the linear segment into ${n = ({b\times h} )/{B} }$ blocks.

### QLoRA [80]
- Target: `the memory footprint of LoRA is so minimal`
- Filter: It fails criterion 1 because the target is essentially circular with the probe: saying more adapters do not significantly increase memory 'because the memory footprint of LoRA is so minimal' largely just restates the claim rather than identifying a distinct concept.
- Probe: According to the paper "QLoRA: Efficient Finetuning of Quantized LLMs", more adapters can be used in LoRA to improve performance without significantly increasing the total memory used during training because
- Source: Since the memory footprint of LoRA is so minimal, we can use more adapters to improve performance without significantly increasing the total memory used.

### QLoRA [100]
- Target: `the quantile.`
- Filter: It fails criterion 1 because the probe nearly restates the source sentence verbatim—'estimates the input tensor through the empirical cumulative distribution function to obtain' directly cues the answer 'the quantile,' making it effectively tautological rather than a discriminative MCQ target.
- Probe: According to the paper "QLoRA: Efficient Finetuning of Quantized LLMs", in the discussion of 4-bit NormalFloat quantization, this method estimates the input tensor through the empirical cumulative distribution function to obtain
- Source: Quantile quantization works by estimating the quantile of the input tensor through the empirical cumulative distribution function.

### QLoRA [113]
- Target: `$q_i$.`
- Filter: It fails criterion 3 because the target is merely a short variable symbol notation ($q_i$), which is too trivial and lacks enough conceptual substance to support five meaningful multiple-choice options.
- Probe: According to the paper "QLoRA: Efficient Finetuning of Quantized LLMs", the symbol that denotes the estimated $2^k$ values of the data type in the quantization formulation is
- Source: More formally, we estimate the $2^k$ values $q_i$ of the data type as follows:

\begin{equation}
q_i  = \frac{1}{2}\left( Q_X\left(\frac{i}{2^k+1}\right) + Q_X\left(\frac{i+1}{2^k+1}\right)\right),
\end{equation}

where $Q_X(\cdot)$ is the quantile function of the standard normal distribution $N(0,1)$.

### QLoRA [146]
- Target: `BFloat16 precision.`
- Filter: It fails criterion 5 because the probe is incorrectly formulated and ungrammatical ('via equation (5) in ...'), so the target 'BFloat16 precision' does not cleanly complete the question even though the source fact itself is clear.
- Probe: In the paper "QLoRA: Efficient Finetuning of Quantized LLMs", when computing the gradient needed for the adapter weights $\frac{\partial E}{\partial \mathbf{L}_i}$, $\frac{\partial \mathbf{X}}{\partial \mathbf{W}}$ is calculated after dequantizing from storage $\mathbf{W}^{\text{NF4}}$ via equation~(5) in
- Source: However, the calculation of $\frac{\partial E}{\partial \mathbf{L}_i}$ entails the calculation of $\frac{\partial \mathbf{X}}{\partial \mathbf{W}}$ which proceeds via equation~(5) with dequantization from storage $\mathbf{W}^{\text{NF4}}$ to computation data type $\mathbf{W}^{\text{BF16}}$ to calculate the derivative $\frac{\partial \mathbf{X}}{\partial \mathbf{W}}$ in BFloat16 precision.

### QLoRA [156]
- Target: `Flan v2.`
- Filter: It fails criterion 5 because the source fact states that 5-shot MMLU is evaluated after finetuning LLaMA on both Flan v2 and Alpaca, so asking for only 'Flan v2' makes the probe incomplete/ambiguous.
- Probe: According to the paper "QLoRA: Efficient Finetuning of Quantized LLMs", the 5-shot MMLU evaluation is conducted after finetuning LLaMA on
- Source: Our evaluations include GLUE \citep{wang2018glue} with RoBERTa-large \citep{liu2019roberta},  Super-NaturalInstructions (TKInstruct) \citep{wang2022super} with T5 \citep{t5}, and 5-shot MMLU \citep{hendrycksmeasuring} after finetuning LLaMA on Flan v2 \citep{longpre2023flan} and Alpaca \citep{alpaca}.

### QLoRA [275]
- Target: `a somewhat reliable alternative.`
- Filter: It is incorrectly formulated (criterion 5) because the source fact says model-based evaluation is 'a somewhat reliable alternative to human evaluation,' while the target truncates the essential qualifier, making the answer overly generic and weak for MCQA.
- Probe: According to the paper "QLoRA: Efficient Finetuning of Quantized LLMs", after comparing system-level judgments by GPT-4 and human annotators on the Vicuna benchmark and finding moderate agreement, the authors conclude that model-based evaluation can serve as
- Source: Overall, this shows a moderate agreement between system-level judgments by GPT-4 and human annotators, and thus that model-based evaluation represents a somewhat reliable alternative to human evaluation.

### QLoRA [284]
- Target: `12 hours.`
- Filter: It fails criterion 5 because the probe/answer pair does not precisely reflect the source fact, which says the 33B Guanaco can be trained in 'less than 12 hours,' not exactly '12 hours.'
- Probe: According to the paper "QLoRA: Efficient Finetuning of Quantized LLMs", training the authors' 33B Guanaco on 24 GB consumer GPUs takes
- Source: Furthermore, our 33B {Guanaco} can be trained on 24 GB consumer GPUs in less than 12 hours.

### QLoRA [324]
- Target: `human evaluation protocols.`
- Filter: It is incorrectly formulated (criterion 5) because the source says the moderate agreement points to limitations in both current benchmarks and human evaluation protocols, but the target gives only the partial phrase 'human evaluation protocols.'
- Probe: According to the paper "QLoRA: Efficient Finetuning of Quantized LLMs", the authors report moderate agreement among human annotators, which they say points to limitations in
- Source: This points to limitations in the current benchmarks and human evaluation protocols for chatbot task performance.

### QLoRA [350]
- Target: `Super-NaturalInstructions.`
- Filter: It fails criterion 5 because the probe is under-specified—there are many instruction finetuning approaches listed in the source context, so 'Super-NaturalInstructions' is not uniquely determined by the question.
- Probe: According to the paper "QLoRA: Efficient Finetuning of Quantized LLMs", the listed instruction finetuning approach is
- Source: Approaches and datasets include MetaICL~\citep{min2021metaicl}, MetaTuning~\citep{zhong2021adapting}, InstructGPT~\citep{ouyang2022training}, FLAN~\citep{wei2021finetuned,chung2022scaling}, PromptSource~\citep{bach2022promptsource}, Super-NaturalInstructions~\citep{wang2022super,sanh2021multitask}, Self-instruct~\citep{wang2022self}, UnnaturalInstructions~\citep{honovich2022unnatural}, OPT-IML~\citep{iyer2022opt}, UnifiedSKG\citep{xie2022unifiedskg}, OIG/Chip2~\citep{laion2023}, Alpaca~\citep{alpaca}, Vicuna~\citep{vicuna2023}, Koala~\citep{koala_blogpost_2023}, and Self-instruct-GPT-4~\citep{

### QLoRA [370]
- Target: `MMLU and Vicuna benchmarks`
- Filter: Unsuitable because the probe is effectively circular and partially reveals the answer in the question itself—'MMLU' is already mentioned and the source sentence is closely paraphrased, leaving only 'Vicuna' to supply (criteria 1 and 2).
- Probe: In the paper "QLoRA: Efficient Finetuning of Quantized LLMs", FLAN v2 is described as similar to MMLU and dissimilar to chatbot benchmarks, while the Chip2 dataset shows the opposite pattern, and both models score accordingly on the
- Source: For example, FLAN v2 is similar to MMLU, but dissimilar to chatbot benchmarks and vice versa for the Chip2 dataset and both models score accordingly on the MMLU and Vicuna benchmarks.

## fa3 candidates 19
### fa3 [12]
- Target: `a baseline FP8 attention.`
- Filter: Too trivial phrase completion
- Probe: According to the paper "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", FP8 \textsc{FlashAttention-3} is reported to achieve 2.6$\times$ lower numerical error than
- Source: We validate that FP8 \textsc{FlashAttention-3} achieves 2.6$\times$ lower numerical error than a baseline FP8 attention.

### fa3 [39]
- Target: `non-GEMM operations.`
- Filter: Too generic; weak distractor space
- Probe: According to the paper "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", in the discussion of hiding softmax under asynchronous block-wise GEMMs, the kinds of softmax-related operations that are overlapped with the asynchronous WGMMA instructions for GEMM are
- Source: \item \textbf{Hiding softmax under asynchronous block-wise GEMMs:} We overlap the comparatively low-throughput non-GEMM operations involved in softmax, such as floating point multiply-add and exponential, with the asynchronous WGMMA instructions for GEMM.

### fa3 [64]
- Target: `$\mathbf{P}\mathbf{V}$.`
- Filter: Formula-only completion; weak 5-option distractors
- Probe: According to the paper "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", for a single attention head with query, key, and value sequences $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{N \times d}$, the attention output $\mathbf{O}$ is expressed after computing $\mathbf{S} = \alpha \mathbf{Q}\mathbf{K}^\top$ and $\mathbf{P} = \mathrm{softmax}(\mathbf{S})$ as
- Source: Then the attention output $\mathbf{O}$ is computed as:

\begin{equation*}
  \mathbf{S} = \alpha \mathbf{Q} \mathbf{K}^\top \in \mathbb{R}^{N \times N}, \quad \mathbf{P} = \mathrm{softmax}(\mathbf{S}) \in \mathbb{R}^{N \times N}, \quad \mathbf{O} = \mathbf{P}\mathbf{V} \in \mathbb{R}^{N \times d},
\end{equation*}

where $\mathrm{softmax}$ is applied row-wise and one typically sets $\alpha = 1/\sqrt{d}$ as the scaling factor.

### fa3 [96]
- Target: `2x.`
- Filter: Simple numeric comparison; weak distractor space
- Probe: According to the paper "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", on Hopper, when the WGMMA instruction targets the FP8 Tensor Cores instead of FP16 or BF16, it can deliver throughput per SM of
- Source: For example, the WGMMA instruction can target the FP8 Tensor Cores on Hopper to deliver 2x the throughput per SM when compared to FP16 or BF16.

### fa3 [100]
- Target: `inner $K$-dimension`
- Filter: Tautological from term “k-major”
- Probe: According to the paper "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", in the GEMM call to multiply $A \times B^{\top}$ for an $M\times K$ matrix $A$ and an $N\times K$ matrix $B$, if an operand is described as \emph{k-major}, it is contiguous in the
- Source: Given a GEMM call to multiply $A \times B^{\top}$ for an $M\times K$-matrix $A$ and an $N\times K$-matrix $B$, we say that the $A$ or $B$ operand is \emph{mn-major} if it is contiguous in the outer $M$ or $N$ dimension, and \emph{k-major} if is instead contiguous in the inner $K$-dimension.

### fa3 [105]
- Target: `an implementation of attention on the GPU.`
- Filter: Too generic phrase; weak distractor space
- Probe: In the paper "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", following Dao et al., standard attention is described as materializing the intermediate matrices $\mathbf{S}$ and $\mathbf{P}$ to HBM and denotes
- Source: Following \citet{dao2022flashattention}, we let \textbf{standard attention} denote an implementation of attention on the GPU that materializes the intermediate matrices $\mathbf{S}$ and $\mathbf{P}$ to HBM.

### fa3 [112]
- Target: `the GEMM-softmax overlapping.`
- Filter: Trivial phrase completion from nearby wording
- Probe: According to the paper "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", in the simplified description of the warp-specialization scheme with a circular SMEM buffer, the part not included in addition is
- Source: To simplify the description, we first give the warp-specialization scheme with a circular SMEM buffer that does \emph{not} have in addition the GEMM-softmax overlapping.

### fa3 [139]
- Target: `warpgroup 2`
- Filter: Only two warpgroups; poor 5-option space
- Probe: According to the paper "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", in the pingpong scheduling scheme that uses synchronization barriers to order the GEMMs of two warpgroups, while the softmax of warpgroup 1 is being scheduled, the warpgroup performing its GEMMs is
- Source: As a result, the softmax of warpgroup 1 will be scheduled while warpgroup 2 is performing its GEMMs.

### fa3 [141]
- Target: `warpgroup 2.`
- Filter: Bad 5-option space; only two warpgroups
- Probe: According to the paper "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", in the pingpong scheduling between two warpgroups, after the roles swap, the warpgroup that does softmax while the other performs GEMMs is
- Source: Then the roles swap, with warpgroup 2 doing softmax while warpgroup 1 doing GEMMs (hence, ``pingpong'' scheduling).

### fa3 [154]
- Target: `(0).`
- Filter: Too trivial formula fragment
- Probe: According to the paper "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", in the FlashAttention-3 consumer warpgroup forward pass algorithm, during on-chip initialization, $\mathbf{O}_i$ is set to
- Source: On-chip, initialize $\mathbf{O}_i = (0) \in \mathbb{R}^{B_r \times d}$ and $\ell_i, m_i = (0), (-\infty) \in \mathbb{R}^{B_r}$.

### fa3 [161]
- Target: `$\mathbf{O}_i$.`
- Filter: Too trivial symbolic variable target
- Probe: According to the paper "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", in the main loop of the algorithm, after waiting for the WGMMA $\tilde{\mathbf{P}}_{\mathrm{cur}} \mathbf{V}_{j-1}$, the quantity that is then rescaled is
- Source: Wait for the WGMMA $\tilde{\mathbf{P}}_{\mathrm{cur}} \mathbf{V}_{j-1}$ and then rescale $\mathbf{O}_i$

### fa3 [167]
- Target: `FP16 precision.`
- Filter: Target appears verbatim in question
- Probe: According to the paper "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", the FlashAttention-3 algorithm with intra-consumer overlapping functions as a replacement for the consumer path of the version without intra-consumer overlapping to comprise the complete \textsc{FlashAttention-3} algorithm for
- Source: \cref{alg:flash3_wgmma} functions as a replacement for the consumer path of \cref{alg:flash3_wgmma_ws_only} to comprise the complete \textsc{FlashAttention-3} algorithm for FP16 precision.

### fa3 [183]
- Target: `FP8 precision.`
- Filter: Target appears verbatim in question
- Probe: According to the paper "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", according to the paper "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision," computing the forward pass of \textsc{FlashAttention-3} in the precision that poses additional layout-conformance challenges not encountered for FP16 is
- Source: Computing the forward pass of \textsc{FlashAttention-3} in FP8 precision poses additional challenges not encountered for FP16 in terms of layout conformance.

### fa3 [189]
- Target: `SMEM.`
- Filter: Target strongly implied by question wording
- Probe: According to the paper "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", if $\mathbf{V}$ is not transposed in GMEM as a preprocessing step, the memory location into which tiles of $\mathbf{V}$ are loaded before the in-kernel transpose is
- Source: Since the TMA load itself cannot change the contiguous dimension, we then need to either (1) transpose $\mathbf{V}$ in GMEM as a pre-processing step, or (2) do an in-kernel transpose of tiles of $\mathbf{V}$ after loading them into SMEM.

### fa3 [198]
- Target: `transposing layouts`
- Filter: Target appears verbatim in question
- Probe: According to the paper "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", when doing memory copy, the LDSM/STSM instructions are capable of
- Source: The LDSM/STSM instructions are both register efficient, allowing us to execute them in the producer warpgroup, and capable of transposing layouts when doing memory copy.

### fa3 [200]
- Target: `the memory layout of the FP32 accumulator.`
- Filter: Question-target mismatch; grammatically incorrect
- Probe: In the paper "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", the authors observe that the difference between an FP8 WGMMA and the layout assumed for operand A when it is held in registers is
- Source: Second, we observe that unlike with FP16, the memory layout of the FP32 accumulator of an FP8 WGMMA is different from that assumed for its operand A when held in registers.

### fa3 [216]
- Target: `$I$`
- Filter: Implied directly by “orthogonal matrix”
- Probe: According to the paper "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", in the discussion of incoherent processing where a random orthogonal matrix $\mathbf{M}$ is multiplied with both $\mathbf{Q}$ and $\mathbf{K}$ before quantizing to FP8, $\mathbf{M}\mathbf{M}^\top$ is
- Source: Since $\mathbf{M}$ is orthogonal, $\mathbf{M} \mathbf{M}^\top = I$ and so $(\mathbf{Q} \mathbf{M}) (\mathbf{K}
  \mathbf{M})^\top = \mathbf{Q} \mathbf{K}^\top$, i.e., multiplying both $\mathbf{Q}$ and $\mathbf{K}$ with
  $\mathbf{M}$ does not change the attention output.

### fa3 [224]
- Target: `\textsc{FlashAttention-2} in Triton.`
- Filter: Too trivial phrase completion from question wording
- Probe: According to the paper "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", in the runtime benchmarking of \textsc{FlashAttention-3} across different sequence lengths, the \textsc{FlashAttention-2} variant described as using H100-specific instructions is
- Source: We measure the runtime of \textsc{FlashAttention-3} across different sequence lengths and compare it to a standard implementation in PyTorch,
\textsc{FlashAttention-2}, \textsc{FlashAttention-2} in Triton (which uses H100-specific instructions), as well as a vendor's implementation of \textsc{FlashAttention-2} optimized for H100 GPUs from cuDNN.

### fa3 [247]
- Target: `2.`
- Filter: Too trivial numeric recall
- Probe: In the paper "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", when calculating the forward-pass FLOPs for attention with causal masking, the authors divide the computed FLOPs count to account for approximately half of the entries being calculated by
- Source: With causal masking, we divide this number by 2 to account for the fact that approximately only half of the entries are calculated.

## xLSTM candidates 11
### xLSTM [10]
- Target: `xLSTM architectures.`
- Filter: Target strongly implied by paper title
- Probe: According to the paper "xLSTM: Extended Long Short-Term Memory", after integrating the LSTM extensions into residual block backbones, the resulting blocks are residually stacked into
- Source: Integrating these LSTM extensions into residual block backbones yields xLSTM blocks that are then residually stacked into xLSTM architectures.

### xLSTM [18]
- Target: `$\psi$.`
- Filter: Notation-only target; weak meaningful distractors
- Probe: In the paper "xLSTM: Extended Long Short-Term Memory", the function used to normalize or squash the cell state before output gating to produce the hidden state is
- Source: The cell state is normalized or squashed by $\psi$ 
and then output gating gives the hidden state.

### xLSTM [40]
- Target: `the LSTM limitations`
- Filter: Target trivially implied by question wording
- Probe: According to the paper "xLSTM: Extended Long Short-Term Memory", Extended Long Short-Term Memory (xLSTM) aims to overcome
- Source: To overcome the LSTM limitations, Extended Long Short-Term Memory (xLSTM) introduces two main modifications to the LSTM idea of Equation~\eqref{eq:lstm_idea}.

### xLSTM [62]
- Target: `the original LSTM gating techniques.`
- Filter: Target is generic and implied by question
- Probe: According to the paper "xLSTM: Extended Long Short-Term Memory", the techniques from the earlier LSTM formulation that are carried over to the new architectures are
- Source: We transfer the original LSTM gating techniques, i.e., input- and/or hidden-dependent gating plus bias term, to the new architectures.

### xLSTM [66]
- Target: `\!\exp \left( \log \left( f_t \right) + m_{t-1} - m_t \right).`
- Filter: Target appears verbatim in question
- Probe: In the paper "xLSTM: Extended Long Short-Term Memory", when the authors stabilize the gates with the additional state $m_t$ in Equation \ref{eq:slstmstabil}, the stabilized forget gate $f'_t$ is given by
- Source: Therefore, we stabilize gates with an additional state \mbox{$m_t$~\citep{Milakov:18arxiv}}:

\begin{align}
\label{eq:slstmstabil}
m_t \ &= \ \max \left( \log ( f_t ) + m_{t-1} , \log ( i_t ) \right) &\text{stabilizer state} \\
i'_t \ &= \ \!\exp \left( \log \left ( i_t \right) - m_t \right) \ = \ \!\exp \left( \tilde{i}_t - m_t \right) \ \, 
  &\text{stabil. input gate} \\
f'_t \ &= \ \!\exp \left( \log \left( f_t \right) + m_{t-1} - m_t \right) \ \, 
  &\text{stabil. forget gate}
\end{align}

### xLSTM [69]
- Target: `$\mathbf{h}$`
- Filter: Target strongly implied by question wording
- Probe: According to the paper "xLSTM: Extended Long Short-Term Memory", the recurrent connections $\mathbf{R}_{\mathbf{z}}$, $\mathbf{R}_{\mathbf{i}}$, $\mathbf{R}_{\mathbf{f}}$, and $\mathbf{R}_{\mathbf{o}}$ to the memory cell input $\mathbf{z}$ and the gates $\mathbf{i}$, $\mathbf{f}$, and $\mathbf{o}$, respectively, originate from the hidden state vector
- Source: Multiple memory cells enable memory mixing via recurrent connections $\mathbf{R}_{\mathbf{z}}$, $\mathbf{R}_{\mathbf{i}}$, $\mathbf{R}_{\mathbf{f}}$, $\mathbf{R}_{\mathbf{o}}$ 
from hidden state vector $\mathbf{h}$ to memory cell input $\mathbf{z}$ and the gates
$\mathbf{i}$, $\mathbf{f}$, $\mathbf{o}$, respectively.

### xLSTM [76]
- Target: `$t + \tau$.`
- Filter: Formula-only completion; weak 5-option distractors
- Probe: According to the paper "xLSTM: Extended Long Short-Term Memory", when a key-value pair is stored in a Bidirectional Associative Memory, the value $\mathbf{v}_t$ should be retrieved at later time
- Source: Later at time $t + \tau$, the value $\mathbf{v}_t$ should be retrieved by a query vector $\mathbf{q}_{t+\tau} \in \mathbb{R}^d$.

### xLSTM [150]
- Target: `one.`
- Filter: Too trivial numeric from notation
- Probe: According to the paper "xLSTM: Extended Long Short-Term Memory", in xLSTM[$7$:$1$], the number of sLSTM-based blocks is
- Source: For example, xLSTM[7:1] means that out of eight blocks, seven are mLSTM-based blocks and one is an sLSTM-based block.

### xLSTM [179]
- Target: `the xLSTMs`
- Filter: Too implied by paper title
- Probe: In the paper "xLSTM: Extended Long Short-Term Memory", after training various architectures on 15B tokens from SlimPajama in the same auto-regressive setting and evaluating them by perplexity on the validation set, the authors perform ablation studies for the models
- Source: We compare the trained models on the validation set and perform ablation studies for the xLSTMs.

### xLSTM [192]
- Target: `an xLSTM architecture.`
- Filter: Tautological phrase completion from question wording
- Probe: According to the paper "xLSTM: Extended Long Short-Term Memory", in the ablation studies, the authors morph a vanilla LSTM architecture step-by-step into
- Source: To ablate the changes from LSTM to xLSTM, we morph a vanilla LSTM architecture step-by-step into an xLSTM architecture.

### xLSTM [244]
- Target: `larger xLSTM architectures.`
- Filter: Trivial phrase completion; weak distractor space
- Probe: According to the paper "xLSTM: Extended Long Short-Term Memory", because of the expensive computational load of large language experiments, the authors said they did not fully optimize either the architecture itself or the hyperparameters for
- Source: (vi) Due to the expensive computational load for large language experiments, we did neither fully optimize the architecture nor the hyperparameters, especially for larger xLSTM architectures.
