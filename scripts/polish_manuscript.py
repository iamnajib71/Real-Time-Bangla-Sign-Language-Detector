"""Apply the final academic house style after the report-driven paper generators.

Run from the repository root. Does not change predictions or numeric evidence.
"""
from pathlib import Path
import re


def polish(path):
    text = path.read_text(encoding="utf-8")
    if r'\bibitem{bdslw401}' in text and 'https://ror.org/05wdbfp45' in text:
        # The editorial revision already has the final house style. Preserve
        # its expanded literature and reviewed affiliation rather than
        # replaying wording replacements written for the earlier draft.
        assert r'\hypersetup{hidelinks' in text
        assert r'\@seccntformat' in text
        print('Preserved final editorial manuscript style, literature and ROR affiliation')
        return
    text = text.replace(r"\hypersetup{colorlinks=true,urlcolor=blue,linkcolor=blue}",
                        r"\hypersetup{hidelinks,pdftitle={Bangla Sign Language Gesture Recognition},pdfauthor={Nazmul Hassan}}")
    text = re.sub(r"\\pagestyle\{fancy\}.*?\\fancyfoot\[C\]\{\\thepage\}",
                  lambda _: r"\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small Bangla Sign Language Gesture Recognition}\fancyfoot[C]{\thepage}", text)
    text = re.sub(r"\\title\{.*?\}\s*\\author\{.*?\}\s*\\date\{.*?\}", lambda _: r"""\title{Bangla Sign Language Gesture Recognition:\\Static Image Classification and Bounded Sentence Recognition}
\author{Nazmul Hassan\\North South University, Dhaka, Bangladesh\\\small ORCID: 0000-0001-6116-017X\\\small \href{mailto:iam.najib@yahoo.com}{iam.najib@yahoo.com}}
\date{\small Preprint}
""".strip(), text, flags=re.S)
    style = r"""\makeatletter
\renewcommand{\@seccntformat}[1]{\csname the#1\endcsname.\quad}
\makeatother
"""
    if r"\@seccntformat" not in text:
        text = text.replace(r"\begin{document}", style + r"\begin{document}")
    setting = r"\renewcommand{\topfraction}{.9}\renewcommand{\textfraction}{.05}\renewcommand{\floatpagefraction}{.85}"
    while text.count(setting) > 1:
        text = text.replace(setting, "", 1)
    replacements = {
        "This revision reports a reproducible reconstruction of a static Bangla hand-sign classifier derived from a 2022 undergraduate thesis.": "This study evaluates static Bangla hand-sign classification and exploratory recognition of a bounded sentence vocabulary, extending an undergraduate thesis system.",
        "A separate current experiment uses": "A separate experiment uses",
        "All results are reproduced measurements, with": "Evaluation uses",
        "The work does not establish continuous language translation, and the paper explicitly separates preserved historical claims from reproducible contemporary evidence.": "The results support constrained recognition experiments; continuous translation and performance with new signers require further evaluation.",
        "This project therefore evaluates static class prediction, rather than claiming a general translator.": "We evaluate static class prediction and a separate, bounded sentence-recognition task.",
        "The original preprint appeared in 2022 on ScienceOpen and was not peer-reviewed~\\cite{hassan}. Its methods narrative and the surviving code differ. This revision preserves that document, identifies the executable method, and adds a traceable code and model release. The present contribution is reproducibility and implementation quality, not a claim of a novel neural architecture or a new record on a public benchmark.": "The original system was described in a ScienceOpen preprint~\\cite{hassan}. Differences between its methods narrative and the surviving implementation motivate an explicit reconstruction of the executable CNN. This study contributes audited data partitions, reproducible model comparisons, class-level error analysis and an exploratory temporal model. The original preprint is retained unchanged; the additional experiments are reported separately.",
        "No claim is made that this older backbone represents the latest architecture, that it has been tuned optimally for Bangla signs, or that its scores are directly comparable with results from other papers.": "Its frozen-backbone configuration provides a controlled, computationally economical transfer-learning baseline.",
        "These distinctions matter when rebuilding an old project. Replacing a historical dataset with a newer one changes both the data distribution and the output task. This revision consequently reports a historical-data comparison and a separate public-data experiment. A higher public-data score, if observed, would not constitute a direct improvement over the 2022 system.": "The historical and public collections have different image distributions and label spaces. We therefore compare architectures within the historical split and report the public-data experiment separately; cross-dataset score differences cannot isolate architectural improvement.",
        "Its 36 numeric folder labels inherit a provisional display mapping from the old interface. The duplicate vowel shown for class 6 in that interface is mapped to O; expert verification remains outstanding. Numeric class identifiers are used in the quantitative tables to avoid hiding this limitation.": "Its 36 numeric folder labels inherit a provisional display mapping from the original interface. The duplicate vowel for class 6 is assigned a provisional display correction; expert verification remains outstanding. Quantitative evaluation retains the original numeric identifiers.",
        "Hand localisation is not implemented: the supplied image or frame must already be a suitable hand crop.": "The static classifier requires a suitable hand crop; automatic hand localisation is outside this image-classification pipeline.",
        r"\section{Reproduction results}": r"\section{Static classification results}",
        "None of these rows is a recovered thesis figure. The historical CNN's measured outcome is retained without replacing it with an aspirational or published number.": "These are additional experimental measurements, reported independently of the original preprint's accuracy statements.",
        "Its exact values are also committed as CSV. Concentration in a small number of columns indicates class collapse or poor discrimination; such outcomes must be reported as observed rather than repaired by relabelling predictions after viewing the test set.": "Raw counts are also available as CSV. Concentration in a small number of prediction columns indicates poor discrimination across the historical classes.",
        r"\section{Software validation and camera-free demonstration}": r"\section{Implementation and software validation}",
        "This distinction prevents a file-processing demonstration from being mistaken for evidence of robust live recognition.": "",
        "Commercial use would require a Deaf-user-led task definition, resolved data permissions and consent, fresh signer/session-disjoint evaluation, novel unsupported utterances, continuous-signing boundary/error analysis and measured call latency. The current exploratory results do not establish these requirements. The repository records an explicit validation roadmap rather than a deployment-readiness claim.": "Deployment evaluation requires Deaf-user-led task design, resolved data permissions and consent, independent signers and sessions, novel unsupported utterances, continuous-signing error analysis and measured end-to-end latency. These remain open research and validation requirements.",
        "Neither dataset permits a verified signer-disjoint partition in the layout used here, and capture-condition dependence can inflate image-level results.": "Neither static-image dataset permits a verified signer-disjoint partition in the layout used here, and capture-condition dependence can inflate image-level results.",
        "The thesis tag preserves the historical files without rewriting Git history. Overleaf users should upload the contents of \\texttt{paper/v2} and compile \\texttt{main.tex}; the project is self-contained.": "KU-BdSL v4 is available from its publisher~\\cite{kudata} under CC BY 4.0. The historical image records and split manifests are preserved in the repository, although original consent documentation is unavailable. Ban-Sign-Sent source clips remain with the original provider; their reuse permissions are unresolved, and those clips are not redistributed.",
        "Version 1 is retained verbatim. This revision adds a reproducible code link, identifies the surviving CNN instead of merging incompatible method descriptions, records new data audits and split manifests, reports only newly reproduced metrics, adds a separate public-data transfer baseline, and documents errors, software validation and ethical limits. It does not reinterpret the original aspirational accuracy statement as a measured result. It does not claim to reproduce the described sparse-autoencoder, depth or MATLAB procedures from unavailable executable evidence. This remains a non-peer-reviewed preprint draft. The author must upload it as a new version of the same ScienceOpen preprint; no new DOI or venue is claimed here.": "The original preprint is unchanged. This version specifies the executable CNN and its differences from the earlier method description; adds audited partitions, model-release links, per-class metrics and error analysis; and reports separate public-data and bounded sentence-recognition experiments. Original accuracy statements are not treated as results of these experiments. Sparse-autoencoder, depth and MATLAB procedures described previously are not reproduced because corresponding executable evidence is unavailable.",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    start = r"\section{Implementation and software validation}"
    end = r"\section{Moving-phrase feasibility experiment}"
    implementation = r"""\section{Implementation and software validation}
The Python implementation provides command-line and PyQt5 interfaces for images, recorded video and cameras. Automated tests cover Otsu preprocessing, tensor contracts, image decoding, Keras prediction and video inference. Synthetic image and video fixtures test software behaviour independently of sign-recognition accuracy. Recorded-file sampling follows media timestamps, and tracking state resets between inputs.

The desktop video-input flow was also tested with a supported clip, a background false accept and a cropped BRAC tutorial greeting. The supported clip was recognised correctly; both unsupported inputs produced incorrect supported captions, including a coffee-related sentence for the greeting. Suggestions required confirmation before transcript insertion. No weights or thresholds were adjusted after these checks. The first two clips were previously evaluated examples, so this test measures integration consistency rather than new recognition accuracy. A separate letter-composition demonstration uses selected correctly classified held-out images; it is not continuous signing.

"""
    if start in text and end in text:
        a, b = text.index(start), text.index(end)
        text = text[:a] + implementation + text[b:]
    text = text.replace("The distinct letter-spelling demonstration decodes a generated video and forms a Bengali sentence from stable isolated-letter predictions with vowel-composition rules. Its examples are deliberately selected correctly predicted held-out images. It proves the text-entry pipeline for those inputs, not spontaneous signing recognition. The code, segment annotations and complete temporal test errors are supplied separately from the static image metrics.", "Segment annotations and temporal errors are reported separately from the static image metrics.")
    text = text.replace(r"\begin{center}\resizebox{\textwidth}{!}{\input{results-table.tex}}\end{center}", r"""\begin{table}[!ht]\centering
\caption{Static classification on fixed image-level test partitions. Historical models share one split; KU-BdSL is a separate task.}\label{tab:static}
\resizebox{\textwidth}{!}{\input{results-table.tex}}
\end{table}""")
    text = text.replace(r"\begin{center}\includegraphics[width=.78\textwidth]{figures/original-confusion.png}\end{center}", r"""\begin{figure}[!ht]\centering
\includegraphics[width=.78\textwidth]{figures/original-confusion.png}
\caption{Historical CNN confusion matrix for 148 test images. Rows denote actual folder IDs and columns denote predicted IDs; entries are raw counts.}\label{fig:historical-confusion}
\end{figure}""")
    text = text.replace(r"\textbf{Historical CNN confusion matrix.} Rows are actual folder IDs and columns are predicted folder IDs. The matrix is generated directly from held-out predictions. Raw counts are also available as CSV.", "Figure~\\ref{fig:historical-confusion} shows held-out predictions; the raw counts are available as CSV.")
    text = text.replace(r"\begin{center}\includegraphics[height=3.6cm]{figures/ku-mobilenet-confusion.png}\end{center}", r"""\begin{figure}[!ht]\centering
\includegraphics[width=.85\textwidth]{figures/ku-mobilenet-confusion.png}
\caption{KU-BdSL MobileNetV2 confusion matrix for 240 test images in 30 grouped classes. Rows are actual IDs and columns predicted IDs.}\label{fig:public-confusion}
\end{figure}""")
    first = r"\begin{minipage}{.45\textwidth}\centering\includegraphics[height=3.2cm]{correct.png}"
    if r"\label{fig:examples}" not in text:
        text = text.replace(first, r"\begin{figure}[!ht]\centering" + "\n" + first)
        last = r"Score 25.5\%.\end{minipage}\hfill"
        text = text.replace(last, last + "\n" + r"\caption{A correct and an incorrect held-out KU-BdSL prediction. Scores are uncalibrated softmax outputs. Images: Jim et al., KU-BdSL v4, CC BY 4.0; resized derivatives.}\label{fig:examples}" + "\n" + r"\end{figure}")
    text = re.sub(r"\n{3,}", "\n\n", text)
    path.write_text(text, encoding="utf-8")
    table_path = path.parent / "results-table.tex"
    table = table_path.read_text(encoding="utf-8")
    for old, new in [("original &", "Historical CNN &"), ("ku-mobilenet &", "KU-BdSL MobileNetV2 &"), ("mobilenet &", "Historical MobileNetV2 &")]:
        table = table.replace(old, new)
    table_path.write_text(table, encoding="utf-8")
    for forbidden in ["Revision draft", "Reproducibility revision draft", "urlcolor=blue", "The author must upload", "Overleaf users should"]:
        assert forbidden not in text, forbidden
    assert text.count(r"\@seccntformat") == 1
    print("Applied academic manuscript style; predictions and metric tables unchanged")


if __name__ == "__main__":
    import sys
    polish(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("paper/v2/main.tex"))
