\# Research Notebook



Predictions written on 2026-10-04, before any experiment was built.

Library: 50 FDA drug labels (data snapshot commit 325f3eb).



\## The question

Can a question-answering system over drug labels tell when its answer

is likely wrong, and act more safely when it is unsure?



Three ways the system can estimate "how sure" it is:

1\. Search score: how well the best label paragraph matches the question.

2\. Agreement: ask the same question 5 times; do the answers agree?

3\. Self-report: the AI states its own confidence ("I'm 80% sure").



\## RQ1: Which confidence signal best separates right from wrong answers?

\*\*H1:\*\* Self-report will be the least trustworthy (overconfident).

Agreement will be the best single signal on normal questions.

\*\*Why:\*\* Self-report is the system grading itself, like trusting a

filter's own error estimate without checking it against real data.

Agreement is measured from the outside, across several answers.



\## RQ2: When search results get worse, do the signals notice?

Test conditions: remove the paragraph with the answer; add misleading

paragraphs; return fewer results.

\*\*H2:\*\* When the right paragraph is removed, the search score will drop,

but agreement and self-report will stay high, because the AI answers

consistently from memory. Consistent does not mean grounded.

\*\*Why:\*\* The AI has seen medical text in training, so it can answer

without a source. Only the search score directly measures the evidence.

\*\*Implication:\*\* Combining signals should work better than any single one.



\## RQ3: Does an assistant that acts on its uncertainty do better?

Compare: (a) simple rule: "if search score is low, say I don't know"

versus (b) an assistant that, when unsure, searches again or checks

another source before deciding.

\*\*H3:\*\* (b) will answer more questions correctly before having to

abstain, but will be slower and cost more per question.

\*\*Why:\*\* Extra steps recover answers that a single search misses, but

every extra step is another AI call.



\## Rules I commit to before running experiments

\- The exam questions are written and frozen before I tune anything.

\- I report every result, including ones that contradict my predictions.

\- I do not remove exam questions after seeing how the system does on them.



\## Log

\- 2026-10-04: Predictions written. 50-label library committed.

