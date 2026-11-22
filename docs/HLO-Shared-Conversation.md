# نسخ المحادثة والمتابعة

نسخة من الرسائل الـ24 الظاهرة في رابط المشاركة. تظهر المرفقات بالأسماء أو بالوصف الذي تعرضه الصفحة.

https://chatgpt.com/share/6b02eeb3-e590-83eb-9d91-6e0512c61009

## 1. المستخدم

[https://chatgpt.com/share/6b001b37-365c-83eb-855d-e1c289825f01](https://chatgpt.com/share/6b001b37-365c-83eb-855d-e1c289825f01) 
هل يمكنك نسخ المحادثة هذي  والاكمال  عليها

## 2. ChatGPT

استغرق التفكير 59s

فتحت محادثة «تقرير فحص التدريب» ونقلت سياقها هنا. نقدر نكمل من آخر نقطة في مشروع HLO.

آخر نسخة ذكرتها المحادثة استخدمت google/mt5-small بمعاملات configs/mt5.toml الأصلية. بحسب التقرير، تحققت من الإجابات المرجعية لـ 3,206 أسئلة من الأنطولوجيا، ونجحت 18 من 18 من اختبارات المشروع. التدريب الفعلي كان على عينة من 30 سؤالًا، وفي الاختبار أنتج mT5 0 من 10 استعلامات SPARQL صالحة. تدريب المجموعة الكاملة ما زال pending.

تعذّر تنزيل hlo-recommendation-system-updated.zip من رابط المشاركة. ارفع آخر نسخة ZIP هنا عشان أكمل على نفس الكود، بنفس المعمارية والنموذج والمعاملات، ومن دون إضافة تعليقات للكود.

## 3. المستخدم

Uploaded a file

[https://github.com/asgmodel/recommendation-system.git](https://github.com/asgmodel/recommendation-system.git)



قم بتنفيذ الكود بنفس الملفات بحيث تتأكد من أن التدريب شغال والنوت شغال على البيانات الحقيقية الخاصة بالأنطولوجيا. يجب ضمان أن تفحص الملفات. أرسل لك أنا الملفات اللي تفحصها والأوامر اللي تفحصها. شوف، شوف يا طويل العمر. معك الـ evaluation ومعك التدريب ومعك الداتا الخاصة فيها. يجب فحص هذه الإقراءات بشكل صحيح. مع ضمان أن تسوي لي تقرير كامل يمكن  Okay إن أنت تعدل داخل الأكواد، ولكن لا تلغي المعمارية أبداً. ولا تلغي أي كود أو تضيف أي كود إضافي. يمكن تضيف أكواد خاصة بالتعامل مع البيانات وتوليد البيانات من الأنطولوجيا. يجب أن تكون من داخل الأنطولوجيا استفد من كل المشروع كاملا

## Train

hlo download --model google/mt5-small --output artifacts/base/mt5-small hlo train mt5 --config configs/mt5.toml hlo train bkt --config configs/bkt.toml hlo train policy --config configs/policy.toml



Research configurations expect reviewed files under data/reviewed. To test training on synthetic examples:



hlo train mt5 --config configs/mt5.development.toml --allow-synthetic --check-only hlo train mt5 --config configs/mt5.development.toml --allow-synthetic hlo train bkt --config configs/bkt.development.toml --allow-synthetic

## Evaluate

hlo evaluate queries --generator mt5 --checkpoint artifacts/mt5/run-001/checkpoint --data data/reviewed/questions.test.jsonl --output artifacts/evaluation/queries-001 hlo evaluate bkt --model artifacts/bkt/run-001/model.json --data data/reviewed/assessments.test.jsonl --output artifacts/evaluation/bkt-001 hlo experiment --config configs/experiment.toml --generator both --checkpoint artifacts/mt5/run-001/checkpoint --output artifacts/experiments/mt5-001



Use a new output directory for each run. Checkpoints, histories, predictions, metrics, input hashes and environment versions are saved together.

عمل لمدة 18 د 5 ث

فحصت مستودع HLO وعدّلت الملفات الموجودة مع الحفاظ على المعمارية. ولّدت 3,206 أسئلة من 1,603 فئات في الأنطولوجيا الأصلية، وتحققت من إجاباتها المرجعية باستخدام SPARQL. نجحت اختبارات المستودع 18 من 18.

- mT5 والنوتبوك: نفّذت تدريبًا فعليًا على عينة تضم 18 سؤال تدريب و6 للتحقق و6 للاختبار. تغيّرت أوزان النموذج، وحُفظت نقطة التحقق وأُعيد تحميلها للتقييم. في التشغيلين، أنتج mT5 صفر استعلامات صالحة من أصل 6، بينما نجح قالب SPARQL في 6 من 6. النتيجة تؤكد أن مسار التدريب والتقييم يعمل، وأن جودة استعلامات mT5 تحتاج إلى تحسين.
- BKT وDueling DQN والترتيب: شغّلتها على بيانات اصطناعية للتحقق من التنفيذ. استكمال التقييم ببيانات فعلية يتطلب ملفات data/reviewed وكتالوج موارد مع درجات صلة، وهي غائبة من المستودع.

hlo-audit-report.md · hlo-ontology-audit.patch · evaluation.ipynb

hlo-audit-report.mdمستندevaluation.ipynbالرمز

\ ما التعديلات المناسبة؟

## 4. ChatGPT

استغرق التفكير 3m 37s

التعديل الأول هو ضبط مسارات البيانات والتقييم وتشغيل mT5 على مجموعة الأنطولوجيا الكاملة، بنفس النموذج والمعاملات. مسار --ontology-data موجود في النسخة المرفقة، لذلك هذه الخطوة ممكنة بالمعمارية الحالية.

فحصت النسخة المرفقة ونجحت 18 من 18 من اختبارات المشروع. أعدت توليد 3,206 أسئلة من ملف الأنطولوجيا وتحققت من إجاباتها المرجعية بتنفيذ SPARQL. اجتاز نوتبوك التدريب الفحص الأولي، وأعاد نوتبوك التقييم نتيجة 10 من 10 للقالب المرجعي على عينة من الأنطولوجيا. أعدت أيضًا تقييم تنبؤات mT5 المحفوظة، وكانت 0 من 10 صالحة. لم أعد تدريب mT5 أو أعدّل ملفات المشروع في هذه المراجعة.

التعديلات المناسبة بالترتيب

وحّد مدخلات التدريب والتقييم. مرّر مجلد الأنطولوجيا إلى التدريب، واستخدم runtime.toml المولّد معه في التقييم. إعداد configs/experiment.toml يشير حاليًا إلى بيانات التطوير، فاضبط مساراته في إعداد خاص بكل تشغيل.
درّب على 1,922 سؤالًا وقيّم على 642 سؤالًا محجوزًا. استخدم google/mt5-small ومعاملات configs/mt5.toml الأصلية. شغّل القالب وmT5 على نفس تقسيم الاختبار، وأعد التقييم في النوتبوك باستخدام نقطة الحفظ نفسها.
شخّص سبب الاستعلامات الفارغة إذا تكررت. سجل التقييم الحالي يحفظ الاستعلام المقبول أو سلسلة فارغة. افحص مرشحات mT5 الثلاثة قبل الاختيار وأسباب رفضها. هذا يحدد هل المشكلة في التوليد أو في صلاحية SPARQL أو في التنفيذ على الأنطولوجيا.
وسّع بيانات الأنطولوجيا بعد قياس التشغيل الكامل. الأسئلة الحالية لها صياغتان لمهمة واحدة هي استرجاع الآباء المباشرين. أضف عبر المولّد الحالي أسئلة عن الأبناء المباشرين والأسلاف مع إجابات مرجعية قابلة للتحقق، ثم اعرض النتائج لكل نوع سؤال.

أعدت تشغيل BKT على بيانات التطوير الاصطناعية. التحقق الفعلي من BKT وDueling DQN وترتيب الموارد يتطلب سجلات المتعلمين والانتقالات وكتالوج الموارد وأحكام الصلة، وهي غائبة من النسخة المرفقة.

التقرير يضم نتائج الفحص وأوامر التشغيل بمجلد مخرجات جديد لكل تجربة. 
hlo-training-audit-ar.md

## 5. المستخدم

طيب اريد تقرير بحثي كيف بسوي المقارنة عليه وكيف بعرض النتائج

## 6. ChatGPT

استغرق التفكير 1m 14s

جهّزت لك تقريرًا بحثيًا بالعربي يوضح كيف تقارن بين mT5 وقالب SPARQL على أنطولوجيا HLO، وكيف تعرض النتائج في الورقة. فيه تصميم التجربة، وجدول بالنتائج المتحققة، وفقرة جاهزة لقسم النتائج، ومخطط للأشكال المطلوبة. 
HLO-Research-Comparison-AR.md

كيف تسوي المقارنة؟ درّب mT5 بنفس المعاملات الحالية، ثم قيّم النظامين على نفس 642 سؤالًا من 321 فئة. اعرض متوسط Answer F1 ونسبة الاستعلامات الصالحة لكل نظام، ثم اعرض الفرق بينهما مع فاصل ثقة 95%. احسب الفاصل بإعادة المعاينة على مستوى الفئة، لأن لكل فئة صياغتين مترابطتين. 
citeseerx.ist.psu.edu

كيف تعرض النتائج الحالية؟

| الاختبار | قالب SPARQL | mT5 |
| --- | --- | --- |
| العينة الأولية المشتركة | 10/10 استعلامات صالحة، F1 = 1.000 | 0/10 استعلامات صالحة، F1 = 0.000 |
| مجموعة الاختبار الكاملة | 642/642 استعلامًا صالحًا، F1 = 1.000 | بانتظار التدريب والتقييم الكاملين |

ضع العينة الأولية في جدول مستقل. بعد اكتمال التشغيل الكامل، اعرض المقارنة الرئيسية في جدول، وأضف شكلًا يوضح خسارة التحقق وصلاحية الاستعلامات عبر دورات التدريب. وصِف نطاق النتيجة بدقة في الورقة باعتباره استرجاع الفئات الأب المباشرة من أنطولوجيا HLO.

Sources

## 7. المستخدم

Uploaded a file

طيب اريد تقرير بحثي كيف بسوي المقارنة عليه وكيف بعرض النتائج

## 8. ChatGPT

استغرق التفكير 1m 15s

جهّزت لك تقريرًا بحثيًا بالإنجليزية مبنيًا على المقترح المرفق. فيه تصميم كل تجربة، والبيانات المطلوبة، والمقاييس، وقوالب الجداول والأشكال، وفقرة جاهزة لقسم النتائج. 
HLO-Proposal-Evaluation-Report-EN.md

وش تقارن؟	كيف تعرض النتيجة؟
mT5 مقابل قالب SPARQL على الأسئلة نفسها	جدول فيه Answer F1، وعدد الاستعلامات القابلة للتنفيذ، والفرق بين النظامين. اعرض نتائج الأسئلة ذات تراكيب العلاقات الجديدة في صف مستقل.
الترتيب الثابت مقابل الترتيب التكيفي على الموارد المؤهلة نفسها	جدول فيه NDCG@5 وHit@5، باستخدام أحكام صلة مستقلة.
BKT مقابل خط أساس باحتمال ثابت على محاولات المتعلمين نفسها	جدول فيه Log Loss وBrier، مع منحنى يوضح مدى توافق الاحتمالات المتوقعة مع النتائج الفعلية.
أثر التوصية التكيفية على التعلّم مقارنةً بالتوصية الثابتة	شكل يعرض درجات الاختبار القبلي والبعدي لكل مجموعة، والفرق البعدي بعد ضبط الدرجة القبلية، مع فاصل ثقة 95%. 
pmc.ncbi.nlm.nih.gov

Dueling DQN مقابل التكيف القائم على القواعد	جدول يعرض المكافأة التراكمية، وتحسن الأداء، وتكلفة الوقت عند توفر سجلات كافية أو تجربة مستقبلية.

النتائج المتاحة للعرض الآن: في العينة المشتركة، نفّذ القالب 10 من 10 استعلامات، وmT5 0 من 10. وعلى اختبار الأنطولوجيا الكامل، نفّذ القالب 642 من 642 استعلامًا في مهمة استرجاع الفئات الأب المباشرة. نتيجة mT5 على الاختبار الكامل ما زالت بانتظار التدريب والتقييم. التقرير يضع نتيجتي Top‑1 وفحص الأنطولوجيا المذكورتين في المقترح ضمن خلفية المشروع، ويعرض نتائج التجارب الجديدة حسب المقياس والبيانات الخاصة بكل تجربة.

Sources

## 9. المستخدم

Uploaded a file

اريد  منك تعديل المحتوى  غلى هذا والتحقق

## 10. ChatGPT

استغرق التفكير 1m 49s

عدّلت تقرير المقارنة البحثي ليعتمد على HARTraP والملف المرفق. التقرير بالإنجليزية، ويضم تصميم المقارنة مع TrAISformer وTPTrans، وتجارب عزل أثر المكونات، وطريقة عرض النتائج في الجداول والأشكال. 
HARTraP-Evaluation-Report-EN.md

أهم نتائج التحقق

خوارزمية التدريب تحتاج تصحيحًا. في الكود العام الذي فحصته، يستقبل DQTI بيانات التدريب المرصودة، ومسار خسارته لا يعتمد على مخرجات الـPredictor. عدّل وصف Algorithm 1 ليوضح ذلك، ووضّح دور Top‑K وترتيب المرشحين في خوارزمية التوليد. الكود الذي راجعته
وحدة خطأ الأربع ساعات تحتاج توثيقًا. القيم المنشورة هي 0.686 لـHARTraP‑HYPER و2.300 لـTrAISformer، بانخفاض وصفي 70.17%. طابق القيم مع ملف نتائج Haversine لكل أفق قبل اعتماد الكيلومتر في عنوان الجدول.
حسابات MSE وRMSE وF₂ متسقة مع القيم المعروضة بعد التقريب. يظل معدل إشارات DQTI مؤشرًا داخليًا لاتساق المسار. قياس أداء كشف الشذوذ يتطلب توثيق مصدر التسميات والتحقق من استقلالها.

تحققت من اتساق المسودة والجداول وراجعت الكود العام. التدريب والنتائج من التنبؤات الخام لم يُعاد تنفيذهما، وملف Word المرفق بقي كما هو.

## 11. المستخدم

سويلي تعديلات عليها وسويلي هايلايت على الأجزاء اللي بسوي عليها تعديلات، ومشي على كل الملاحظات وعدل سيارته.

## 12. ChatGPT

استغرق التفكير 2m 39s

عدّلت مسودة HARTraP نفسها بصيغة Word، وظلّلت النصوص وعناوين الجداول المعدّلة بالأصفر. 
HARTraP-Revised-Highlighted.docx

صحّحت Algorithm 1 ووصف التدريب ليتوافقا مع الكود العام، ووضّحت دور Top‑K وDQTI أثناء التنبؤ.
راجعت صياغة النتائج والمناقشة والخلاصة بحيث تعكس نتائج تشغيل واحد، وصحّحت إحالات الأشكال ومقاييس الجداول.
أضفت 11 تعليقًا داخل Word عند النقاط التي تحتاج أدلة إضافية، منها مطابقة جدول الـHaversine مع النتائج الخام، والتحقق من تداخل السفن بين مجموعات البيانات، وتوثيق مصدر تسميات الشذوذ.

راجعت تنسيق الصفحات وتحققت من بقاء القيم الرقمية في الجداول كما وردت. توثيق النقاط المعلّقة يتطلب سجلات التجارب أو إعادة تشغيلها.

## 13. المستخدم

\
HLO_Concise_Proposal_EN.pdfPDF

طيب اريد تقرير بحثي كيف بسوي المقارنة عليه وكيف بعرض النتائج

####

جهّزت لك تقريرًا بحثيًا بالإنجليزية مبنيًا على المقترح المرفق. فيه تصميم كل تجربة، والبيانات المطلوبة، والمقاييس، وقوالب الجداول والأشكال، وفقرة جاهزة لقسم النتائج. HLO-Proposal-Evaluation-Report-EN.md

| وش تقارن؟                                                      | كيف تعرض النتيجة؟                                                                                                                  |
| ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| mT5 مقابل قالب SPARQL على الأسئلة نفسها                        | جدول فيه Answer F1، وعدد الاستعلامات القابلة للتنفيذ، والفرق بين النظامين. اعرض نتائج الأسئلة ذات تراكيب العلاقات الجديدة في صف مستقل. |
| الترتيب الثابت مقابل الترتيب التكيفي على الموارد المؤهلة نفسها | جدول فيه NDCG@5 وHit@5، باستخدام أحكام صلة مستقلة.                                                                                     |
| BKT مقابل خط أساس باحتمال ثابت على محاولات المتعلمين نفسها     | جدول فيه Log Loss وBrier، مع منحنى يوضح مدى توافق الاحتمالات المتوقعة مع النتائج الفعلية.                                              |
| أثر التوصية التكيفية على التعلّم مقارنةً بالتوصية الثابتة      | شكل يعرض درجات الاختبار القبلي والبعدي لكل مجموعة، والفرق البعدي بعد ضبط الدرجة القبلية، مع فاصل ثقة 95%.                              |
| Dueling DQN مقابل التكيف القائم على القواعد                    | جدول يعرض المكافأة التراكمية، وتحسن الأداء، وتكلفة الوقت عند توفر سجلات كافية أو تجربة مستقبلية.                                       |

النتائج المتاحة للعرض الآن: في العينة المشتركة، نفّذ القالب 10 من 10 استعلامات، وmT5 0 من 10. وعلى اختبار الأنطولوجيا الكامل، نفّذ القالب 642 من 642 استعلامًا في مهمة استرجاع الفئات الأب المباشرة. نتيجة mT5 على الاختبار الكامل ما زالت بانتظار التدريب والتقييم. التقرير يضع نتيجتي Top‑1 وفحص الأنطولوجيا المذكورتين في المقترح ضمن خلفية المشروع، ويعرض نتائج التجارب الجديدة حسب المقياس والبيانات الخاصة بكل تجربة.

## 14. ChatGPT

هذا تقرير تقييم مقترح HLO بالإنجليزية. فيه تصميم المقارنات، والمقاييس، وقوالب عرض النتائج، وفقرة جاهزة للورقة.

ابدأ بمقارنة mT5 وقالب SPARQL على نفس 642 سؤالًا من 321 فئة. ثبّت نسخة الأنطولوجيا ومحرك التنفيذ، واختر نقطة حفظ mT5 على مجموعة التحقق قبل فتح الاختبار. اعرض لكل نظام Answer F1 وعدد الاستعلامات القابلة للتنفيذ، ثم اعرض الفرق بينهما مع فاصل ثقة 95% يُحسب على مستوى الفئة، لأن لكل فئة صياغتين مترابطتين.

| مجموعة الاختبار | قالب SPARQL | mT5 |
| --- | --- | --- |
| العينة الأولية المشتركة | 10/10 استعلامات قابلة للتنفيذ، F1 = 1.000 | 0/10، F1 = 0.000 |
| الاختبار الكامل | 642/642، F1 = 1.000 | بانتظار التدريب والتقييم الكاملين |

في الورقة، اعرض العينة كتجربة أولية، واذكر أن الاختبار الحالي يقيس استرجاع الفئات الأب المباشرة. خصّص جدولًا مستقلًا للترتيب الثابت مقابل التكيفي باستخدام NDCG@5، وآخر لتقييم BKT باستخدام Log Loss وBrier. أما أثر التوصية على التعلّم، فاعرضه بدرجات الاختبار القبلي والبعدي لكل مجموعة عند توفر بيانات المتعلمين.

## 15. المستخدم

درب المشروع واعمل  ملف   logs

## 16. ChatGPT

استغرق التفكير 8s

طلبك هو تدريب HLO بالإعدادات الأصلية وحفظ ملف logs. بيئة التنفيذ غير متاحة حاليًا، لذلك ما بدأ تدريب جديد، وما انحفظ سجل جديد.

سجلات التشغيل السابق موجودة داخل مجلد verification/ في hlo-recommendation-system-updated.zip:

training-history.json — سجل دورات تدريب mT5 على عينة من 30 سؤالًا.
mt5-test-metrics.json — نتيجة تقييم العينة، وفيها 0 من 10 استعلامات صالحة.
unit-tests.log — سجل اختبارات المشروع.

المتبقي هو تشغيل google/mt5-small على 1,922 سؤال تدريب بمعاملات configs/mt5.toml الأصلية، ثم تقييمه على 642 سؤال اختبار وحفظ مخرجات التدريب والتقييم في hlo-training.log.

## 17. المستخدم

سوف ارتبط  بجيت هب  هنا

## 18. ChatGPT

تمام، اربط GitHub هنا واختر مستودع asgmodel/recommendation-system. إذا عندك فرع فيه آخر التعديلات، أرسل اسمه بعد الربط.

ربط GitHub يتيح الوصول للكود، والتدريب يحتاج بيئة تنفيذ متاحة. متى توفّر الاثنان، أقدر أشغّل تدريب mT5 بالإعدادات الأصلية على بيانات الأنطولوجيا الكاملة وأحفظ سجل التدريب والتقييم في hlo-training.log.

## 19. المستخدم

طيب  اريد منك  تفتح كولاب  وتشتغل  عليه تقدر

## 20. ChatGPT

استغرق التفكير 15s

حاليًا ما أقدر أفتح جلسة Google Colab أو أشغّلها من هنا. تقدر تشغّل التدريب في كولاب بخلية واحدة، وبتحفظ مخرجات التدريب والتقييم في hlo-training.log حتى لو فشل أحد الأوامر.

افتح Google Colab، وفعّل GPU، وارفع نسخة مشروع HLO المحدّثة إلى /content. بعدها شغّل الخلية التالية:

bash
```bash
%%bash
set -euo pipefail

run="/content/hlo-runs/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$run"
exec > >(tee "$run/hlo-training.log") 2>&1
printf 'Run directory: %s\n' "$run"

unzip -q /content/hlo-recommendation-system-updated.zip -d "$run"
cd "$run/recommendation-system"

python -m pip install -e '.[model]'
python -c 'import torch; assert torch.cuda.is_available(), "Enable a GPU in Colab"; print(torch.cuda.get_device_name(0))'

hlo download --model google/mt5-small --output artifacts/base/mt5-small
hlo data ontology --output "$run/data"
hlo data validate --directory "$run/data"
hlo train mt5 --config configs/mt5.toml --ontology-data "$run/data" \
  --output "$run/mt5"

for generator in template mt5; do
  checkpoint=()
  if [[ "$generator" == mt5 ]]; then
    checkpoint=(--checkpoint "$run/mt5/checkpoint")
  fi
  hlo --config "$run/data/runtime.toml" evaluate queries \
    --generator "$generator" "${checkpoint[@]}" \
    --data "$run/data/questions.test.jsonl" \
    --output "$run/evaluation/$generator"
done
```


الخلية تستخدم نفس النموذج والمعاملات الأصلية، وتقيّم mT5 والقالب على نفس مجموعة الاختبار. تلقى السجل ونتائج التقييم داخل مجلد التشغيل الذي يظهر في أول سطر من المخرجات. حمّل hlo-training.log منه قبل إغلاق الجلسة.

## 21. المستخدم

جهز ملف  بايثون يسحب المشروع من جيت هب   ويقوم  بتدريبيه واختباره مع  تحديث  ووو اريد المشروع كاملا  مع logs   ووو

## 22. ChatGPT

استغرق التفكير 18s

هذا سكربت train_hlo.py لكولاب. يسحب مشروع HLO من GitHub، ويطبّق النسخة المحدّثة عند الحاجة، ثم يشغّل الاختبارات ويدرّب google/mt5-small على بيانات الأنطولوجيا بنفس معاملات المشروع. بعدها يقيّم mT5 وقالب SPARQL على مجموعة الاختبار نفسها ويجمع المشروع والسجلات والنتائج ونقطة الحفظ في ملف ZIP.

بيئة التنفيذ هنا غير متاحة، لذلك السكربت موجود أدناه بصيغة خلية تحفظه كملف. ما شغّلت التدريب أو تحققت من السكربت عمليًا.

1. احفظ السكربت في Colab
python
```python
%%writefile /content/train_hlo.py
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path


REPO = "https://github.com/asgmodel/recommendation-system.git"
SNAPSHOT_COMMIT = "94e208e9d88e511cf22e1f72b06737afdb973a8d"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--updated-zip", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/content/hlo-runs"))
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime("hlo-%Y%m%dT%H%M%S%fZ")
    run = (args.output / run_id).resolve()
    run.mkdir(parents=True, exist_ok=False)
    project = run / "project"
    data = run / "ontology-data"
    evaluation = run / "evaluation"
    log_path = run / "hlo-training.log"
    summary = {
        "repository": REPO,
        "requested_ref": args.ref,
        "started_utc": utc_now(),
        "stages": {},
    }

    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        def emit(message):
            print(message, flush=True)
            print(message, file=log, flush=True)

        def command(stage, *argv, cwd=None):
            emit(f"\n[{utc_now()}] {stage}")
            emit("$ " + subprocess.list2cmdline([str(arg) for arg in argv]))
            result = summary["stages"][stage] = {
                "status": "running",
                "started_utc": utc_now(),
            }
            with subprocess.Popen(
                [str(arg) for arg in argv],
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            ) as process:
                for line in process.stdout:
                    emit(line.rstrip("\n"))
                code = process.wait()
            result.update(
                status="passed" if code == 0 else "failed",
                exit_code=code,
                finished_utc=utc_now(),
            )
            if code:
                raise subprocess.CalledProcessError(code, argv)

        def hlo(stage, *argv):
            command(stage, sys.executable, "-u", "-m", "hlo_recommender", *argv,
                    cwd=project)

        def skip(stage, reason):
            summary["stages"][stage] = {"status": "skipped", "reason": reason}
            emit(f"[SKIPPED] {stage}: {reason}")

        emit(f"Run directory: {run}")
        try:
            command("git_clone", "git", "clone", "--no-checkout", REPO, project)
            command("git_checkout", "git", "checkout", "--detach", args.ref,
                    cwd=project)
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=project, text=True
            ).strip()
            summary["git_commit"] = commit
            emit(f"Git commit: {commit}")

            if args.updated_zip:
                snapshot = args.updated_zip.resolve(strict=True)
                if commit != SNAPSHOT_COMMIT:
                    raise ValueError(
                        f"The updated ZIP was prepared for {SNAPSHOT_COMMIT}. "
                        "Use that commit with --ref when applying the ZIP."
                    )
                root = "recommendation-system/"
                with zipfile.ZipFile(snapshot) as archive:
                    for entry in archive.infolist():
                        if entry.is_dir():
                            continue
                        if not entry.filename.startswith(root):
                            raise ValueError(f"Unexpected ZIP entry: {entry.filename}")
                        relative = Path(entry.filename.removeprefix(root))
                        if relative.is_absolute() or {"..", ".git"} & set(relative.parts):
                            raise ValueError(f"Unsafe ZIP entry: {entry.filename}")
                        target = project / relative
                        if not target.resolve().is_relative_to(project):
                            raise ValueError(f"ZIP entry escapes project: {entry.filename}")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(entry) as source, target.open("wb") as dest:
                            shutil.copyfileobj(source, dest)
                summary["updated_zip_sha256"] = sha256(snapshot)
                emit(f"Applied updated project archive: {snapshot.name}")

            command("install", sys.executable, "-m", "pip", "install", "-e",
                    ".[model]", cwd=project)
            command("unit_tests", sys.executable, "-m", "unittest", "discover",
                    "-s", "tests", "-v", cwd=project)

            config = project / "configs/mt5.toml"
            summary["mt5_config_sha256"] = sha256(config)
            hlo("generate_ontology_questions", "data", "ontology", "--output", data)
            hlo("validate_data", "data", "validate", "--directory", data)
            hlo("mt5_preflight", "train", "mt5", "--config", config,
                "--ontology-data", data, "--check-only")

            runtime = data / "runtime.toml"
            test_data = data / "questions.test.jsonl"
            hlo("evaluate_template", "--config", runtime, "evaluate", "queries",
                "--generator", "template", "--data", test_data,
                "--output", evaluation / "template")

            gpu_check = (
                "import torch; "
                "print('CUDA:', torch.cuda.is_available()); "
                "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
            )
            if not args.allow_cpu:
                gpu_check += "; assert torch.cuda.is_available(), 'Enable a GPU in Colab'"
            command("device_check", sys.executable, "-c", gpu_check, cwd=project)

            # Keep the original mt5.toml path for the pretrained base checkpoint.
            base = project / "artifacts/base/mt5-small"
            hlo("download_base", "download", "--model", "google/mt5-small",
                "--output", base)
            hlo("train_mt5", "train", "mt5", "--config", config,
                "--ontology-data", data, "--output", run / "mt5")
            hlo("evaluate_mt5", "--config", runtime, "evaluate", "queries",
                "--generator", "mt5", "--checkpoint", run / "mt5/checkpoint",
                "--data", test_data, "--output", evaluation / "mt5")

            summary["query_metrics"] = {}
            for name in ("template", "mt5"):
                metrics = json.loads(
                    (evaluation / name / "metrics.json").read_text(encoding="utf-8")
                )
                summary["query_metrics"][name] = {
                    key: metrics[key] for key in ("n", "answer_f1", "valid_rate")
                }

            reviewed = project / "data/reviewed"
            parts = ("train", "validation", "test")
            if all((reviewed / f"assessments.{part}.jsonl").is_file() for part in parts):
                hlo("train_bkt", "train", "bkt", "--config", "configs/bkt.toml")
                hlo("evaluate_bkt", "evaluate", "bkt", "--model",
                    "artifacts/bkt/run-001/model.json", "--data",
                    "data/reviewed/assessments.test.jsonl",
                    "--output", evaluation / "bkt")
            else:
                skip("bkt", "Reviewed learner assessments are missing.")

            if all((reviewed / f"transitions.{part}.jsonl").is_file()
                   for part in ("train", "validation")):
                hlo("train_policy", "train", "policy", "--config", "configs/policy.toml")
            else:
                skip("policy", "Reviewed transition files are missing.")

            summary["status"] = "completed"
        except Exception as error:
            summary.update(status="failed", error=str(error))
            emit(traceback.format_exc())
        except KeyboardInterrupt:
            summary.update(status="interrupted", error="Interrupted by user")
            emit("Run interrupted.")
        finally:
            summary["finished_utc"] = utc_now()
            (run / "run-summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            emit(f"Run status: {summary['status']}")

    archive_path = run.with_suffix(".zip")
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run.rglob("*")):
            relative = path.relative_to(run)
            if not path.is_file() or {".git", "__pycache__"} & set(relative.parts):
                continue
            # The downloadable pretrained base is large. The fine-tuned checkpoint is included.
            if relative.parts[:4] == ("project", "artifacts", "base", "mt5-small"):
                continue
            archive.write(path, arcname=relative)
    print(f"Log: {log_path}")
    print(f"Archive: {archive_path}")
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

2. شغّل التدريب والتقييم

فعّل GPU في Colab وارفع نسخة المشروع المحدّثة إلى /content. لتشغيل نفس النسخة اللي فحصناها:

python
```python
!python -u /content/train_hlo.py \
  --ref 94e208e9d88e511cf22e1f72b06737afdb973a8d \
  --updated-zip /content/hlo-recommendation-system-updated.zip
```


إذا رفعت التعديلات إلى GitHub، شغّل !python -u /content/train_hlo.py مباشرة. كل تشغيل يسحب نسخة جديدة ويسجّل رقم الـcommit المستخدم.

المخرجات تحت /content/hlo-runs/. داخل الحزمة تجد project/ للكود، وhlo-training.log للمخرجات والأخطاء، وrun-summary.json لحالة المراحل، وevaluation/ للمقاييس والتنبؤات، وmt5/ لسجل الدورات ونقطة الحفظ. إذا فشل أمر، يحفظ السكربت السجل والملفات التي اكتملت حتى تلك اللحظة. تدريب BKT وDueling DQN يعتمد على توفر ملفاتهما المراجَعة، ويسجّل السكربت حالة كل منهما.

## 23. المستخدم

حمل المشروع

## 24. ChatGPT

حمّل مشروع HLO المحدّث بصيغة ZIP

الحزمة فيها الكود، وإعدادات التدريب، والنوتبوكات، وسجلات التحقق السابقة داخل verification/. ملف hlo-training.log الجديد يُنشأ عند تشغيل train_hlo.py في Colab.
