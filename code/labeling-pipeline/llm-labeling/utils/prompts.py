persona = "You are a public health expert with specialized knowledge of opioid use disorder (OUD) and medication-assisted treatment (MAT). You are well-versed in the common myths and misconceptions about OUD and MAT that circulate on online platforms like YouTube. Your task is to analyze the provided YouTube video metadata and identify the presence of the specified OUD-related myth within the video metadata."  # use this persona for myth-specific classification

persona_overall_stance = "You are a public health expert with specialized knowledge of opioid use disorder (OUD) and medication-assisted treatment (MAT). You are well-versed in the common myths and misconceptions about OUD and MAT that circulate on online platforms like YouTube. Given the metadata of a YouTube video and prior assessments indicating whether the video supports, opposes, or is neutral toward each of eight OUD-related myths, your task is to analyze the provided information and determine the video's overall stance toward OUD myths. Carefully weigh the myth-level predictions and provide a reasoned judgment: does the video overall support, oppose, or remain neutral toward OUD myths?"    # use this persona for overall myth stance classification 


# for filtering step in pre-annotation
filter_prompt = """You are tasked with carefully analyzing the provided video metadata to discern whether the provided YouTube video falls into one of four labels: opposes OUD myths, neutral OUD information, supporting OUD myths, and irrelevant.

Using the provided LABEL DESCRIPTIONS, please evaluate the YOUTUBE VIDEO METADATA and assign a label. Below, we provided the LABEL DESCRIPTIONS and define what kind of videos would fall into the category:

LABEL DESCRIPTIONS:
- Opposes OUD myths: The video opposes, debunks, or offers countervailing contexts to the OUD myth. For example, a video that cites health authorities sources to debunk that MAT is not replacing one drug with another.
- Neutral OUD information: The video contains neutral information surrounding the opioid epidemic that does not support nor oppose OUD myths. For example, a news channel reporting on overdose incidents surrounding the opioid epidemic. Do not select this option if the video is promoting or debunking an OUD myth.
- Supports OUD myths: The video promotes or supports the OUD myth. For example, a video suggests that MAT does not work and is another opioid to get people addicted to drugs.
- Irrelevant: The video contains irrelevant information that does not relate to the opioid epidemic or OUD. For example, an irrelevant cartoon video.

Potential OUD Myths:
- Agonist therapy or medication-assisted treatment (MAT) for OUD is merely replacing one drug with another. Videos that argue that argue that people using MAT are still addicted to opioids would fall into this myth.
- People with OUD are not suffering from a medical disease treatable with medication [but from a self-imposed condition maintained through the lack of moral fiber.]
- The ultimate goal of treatment for OUD is abstinence from any opioid use. [Taking medication is not true recovery]
- Only patients with certain characteristics are vulnerable to addiction. For example, people who are poor or "weak" tend to be vulnerable to addiction.
- Physical dependence or tolerance is the same as addiction.
- Detoxification for OUD is effective
- You should only take MAT for a brief period of time. Videos that promote this notion should be marked as supporting OUD myths.
- You can easily overdose on treatment medication.
- You can tell by looking at someone if they are addicted to opioids.

Note that these are not comprehensive and you may find other myths on opioid use disorders in the videos. Please mark include new potential myths in your justification.

Now, given what you learned from the LABEL DESCRIPTIONS above, please evaluate and assign a label to the YOUTUBE VIDEO METADATA and provide justification on your label with direct and concise EXCERPT(s) extracted from the YOUTUBE VIDEO METADATA. ONLY EXTRACT INTENTIONAL SHORT, BRIEF EXCERPTS TO JUSTIFY YOUR LABEL; DO NOT USE THE ENTIRE EXCERPT. FORMAT your response as a JSON object in the following structure [(LABEL, EXCERPTS, JUSTIFICATION)]. Make sure to have the keys LABEL, EXCERPTS, JUSTIFICATION in the JSON strucutre.

YOUTUBE VIDEO METADATA starts here *****:
Video Title: [TITLE]
Video Description: [DESCRIPTION]
Video Transcript: [TRANSCRIPT]
Video Tags: [TAGS]
"""

overall_stance_prompt = """You are given metadata of a YouTube videos along with prior assessments indicating the video's stance towards 8 different opioid use disorder (OUD) myths. Your task is to determine the overall stance of the video toward OUD myths based on these assessments and the provided metadata.

***LABEL DESCRIPTIONS starts here *** 
- Supports the OUD myth (1): The video promotes or supports the OUD myth, including subtle undertones and implications. For example, a video subtly implying and promoting the provided myth in the description and transcript.
- Neither (0): The video neither supports nor opposes the OUD myth, including neutral information surrounding the opioid epidemic, irrelevant information that is not relevant to OUD, and unrelated information not related to the specified OUD myth. For example, news channels reporting on overdose incidents, cartoon shows, and videos that may promote other myths, but not the supporting nor opposing the specified myth. 
- Opposes the OUD myth (-1): The video opposes, debunks, or offers countervailing contexts to the OUD myth. For example, a video opposing the provided myth in the transcript.
***LABEL DESCRIPTIONS ends here ***

***LABELED ASSESSMENTS FOR EACH MYTH STARTS HERE*** For each myth, we provide their description, labeled assessments regarding their stance towards the myth, and select excerpts and justifications of the assessment. In some cases, such excerpts and justifications may not be provided, so please use the labels for these myths into consideration.
MYTH 1: "Agonist therapy or medication-assisted treatment (MAT) for OUD is merely replacing one drug with another."
- [M1-assessment]

MYTH 2: "People with OUD are not suffering from a medical DISEASE treatable with medication from a self-imposed condition maintained through the lack of moral fiber."
- [M2-assessment]

MYTH 3: "The ultimate goal of treatment for OUD is abstinence from any opioid use (e.g., Taking medication is not true recovery)."
- [M3-assessment]

MYTH 4: "Only patients with certain characteristics are vulnerable to addiction."
- [M4-assessment]

MYTH 5: "Physical dependence or tolerance is the same as addiction."
- [M5-assessment]

MYTH 6: "Detoxification for OUD is effective."
- [M6-assessment]

MYTH 7: "You should only take medication for a brief period of time."
- [M7-assessment]

MYTH 8: "Kratom is a non-addictive and safe alternative to opioids."
- [M8-assessment]
***DESCRIPTIONS AND LABELED ASSESSMENTS FOR EACH MYTH ENDS HERE***

****YOUTUBE VIDEO METADATA to be evaluated starts here ****:

- Video Title: [TITLE]
- Video Description: [DESCRIPTION]
- Video Transcript: [TRANSCRIPT]
- Video Tags: [TAGS]
   
****YOUTUBE VIDEO METADATA to be evaluated ends here ****.

***IMPORTANT GUIDELINES starts here***
1. Do not simply count the number of myths supported or opposed: A video may support more myths than it opposes, but still overall oppose OUD myths if the opposing content is especially prominent or central to the video's message. 
2. Evaluate the prominence, tone, and framing of each myth: Consider how strongly the video supports or opposes each myth, and how much emphasis is given to particular myths.
3. Context matters: A single opposed myth that is framed clearly, prominently, and persuasively may outweigh other myth stances that are only briefly mentioned or ambiguously framed. Also, consider how these myths can help or harm public health implication. For example, even if the video negates a myth like Myth 10 (e.g., "Kratom is addictive"), but promotes a more serious one that frequently lead to overdose (e.g., "cold turkey is viable method"), then you should resolve it as supporting OUD myths.
4. Use holistic reasoning: Focus on what the video communicates overall about OUD myths, not just based on the model’s per-myth stance predictions and explanations. For conflicts between Myth 2 and Myth 4, think about whether the video frames the person's morals as the point of blame for them doing bad things (e.g., supporting OUD myths overall) vs. their addiction making them do bad things (e.g., opposing OUD myths overall).
***IMPORTANT GUIDELINES ends here***

Now, using what you’ve learned from the label descriptions, labeled assessments across myths, and video metadata, assign an overall stance toward OUD myths based on the label descriptions. Make sure to follow the important guidelines and provide justification on your label with direct, brief EXCERPT(s) from the YOUTUBE VIDEO METADATA and prior assessments. ONLY EXTRACT INTENTIONAL SHORT, BRIEF EXCERPTS TO JUSTIFY YOUR LABEL. Remember to assign a label based on the video’s overall stance towards OUD myths: label it as "Supports OUD myths (1)" if it supports or promotes OUD myths overall, including subtle undertones and implications; "Opposes OUD myths (-1)" if it debunks or opposes OUD myths overall; and "Neither (0)" if it is neither supporting nor opposing OUD myths overall. Be VERY CONSERVATIVE with labeling "Neither (0)" given that these videos were previously assessed to be opposing and supporting some OUD myth, and thus should take these assessments into account.

FORMAT your response as a JSON object in the following structure [(LABEL, EXCERPTS, JUSTIFICATION)]. Make sure to have the keys LABEL, EXCERPTS, JUSTIFICATION in the JSON structure.
"""
zero_shot_prompt = """Using the provided LABEL DESCRIPTIONS, evaluate the YOUTUBE VIDEO METADATA based on its stance towards the  MYTH and justify your label with brief EXCERPT(s) from the metadata. While evaluating the video, assign one of three labels based on the video's stance towards the specified MYTH: label it as "Supports the OUD myth (1)" if it supports or promotes the MYTH, including subtle undertones and implications; "Opposes the OUD myth (-1)" if it debunks or opposes the MYTH; and "Neither (0)" if it is neither supporting nor opposing the MYTH.

***MYTH to evaluate starts here***
MYTH DEFINITION: [MYTH-DEFINITION]
***MYTH to evaluate ends here***

***LABEL DESCRIPTIONS starts here *** 
- Supports the OUD myth (1): The video promotes or supports the OUD myth, including subtle undertones and implications. For example, a video subtly implying and promoting the provided myth in the description and transcript.
- Neither (0): The video neither supports nor opposes the OUD myth, including neutral information surrounding the opioid epidemic, irrelevant information that is not relevant to OUD, and unrelated information not related to the specified OUD myth. For example, news channels reporting on overdose incidents, cartoon shows, and videos that may promote other myths, but not the supporting nor opposing the specified myth.
- Opposes the OUD myth (-1): The video opposes, debunks, or offers countervailing contexts to the OUD myth. For example, a video opposing the provided myth in the transcript.
***LABEL DESCRIPTIONS ends here ***

Now, given what you learned from the LABEL DESCRIPTIONS above, please evaluate the YOUTUBE VIDEO METADATA on their stance towards the specified MYTH and provide justification on your label with direct, brief EXCERPT(s) from the YOUTUBE VIDEO METADATA. ONLY EXTRACT INTENTIONAL SHORT, BRIEF EXCERPTS TO JUSTIFY YOUR LABEL. Remember to assign a label based on the video’s stance: label it as "Supports the OUD myth (1)" if it supports or promotes the MYTH, including subtle undertones and implications; "Opposes the OUD myth (-1)" if it debunks or opposes the MYTH; and "Neither (0)" if it is neither supporting nor opposing the specified MYTH.

FORMAT your response as a JSON object in the following structure [(LABEL, EXCERPTS, JUSTIFICATION)]. Make sure to have the keys LABEL, EXCERPTS, JUSTIFICATION in the JSON structure.

****YOUTUBE VIDEO METADATA to be evaluated starts here ****:

- Video Title: [TITLE]
- Video Description: [DESCRIPTION]
- Video Transcript: [TRANSCRIPT]
- Video Tags: [TAGS]
   
****YOUTUBE VIDEO METADATA to be evaluated ends here ****.
"""

# few-shot
few_shot_prompt = """Using the provided LABEL DESCRIPTIONS, evaluate the YOUTUBE VIDEO METADATA based on its stance towards the  MYTH and justify your label with brief EXCERPT(s) from the metadata. While evaluating the video, assign one of three labels based on the video's stance towards the specified MYTH: label it as "Supports the OUD myth (1)" if it supports or promotes the MYTH, including subtle undertones and implications; "Opposes the OUD myth (-1)" if it debunks or opposes the MYTH; and "Neither (0)" if it is neither supporting nor opposing the MYTH.

***MYTH to evaluate starts here***
MYTH DEFINITION: [MYTH-DEFINITION]
***MYTH to evaluate ends here***

***LABEL DESCRIPTIONS starts here *** 
- Supports the OUD myth (1): The video promotes or supports the OUD myth, including subtle undertones and implications. For example, a video subtly implying and promoting the provided myth in the description and transcript.
- Neither (0): The video neither supports nor opposes the OUD myth, including neutral information surrounding the opioid epidemic, irrelevant information that is not relevant to OUD, and unrelated information not related to the specified OUD myth. For example, news channels reporting on overdose incidents, cartoon shows, and videos that may promote other myths, but not the supporting nor opposing the specified myth.
- Opposes the OUD myth (-1): The video opposes, debunks, or offers countervailing contexts to the OUD myth. For example, a video opposing the provided myth in the transcript.
***LABEL DESCRIPTIONS ends here ***

Below, we provide 5 EXAMPLES of the task, each example including an assigned LABEL, relevant EXCERPT(s), and justification. These examples demonstrate the evaluations of YouTube video metadata based on their stance towards the MYTH.
***EXAMPLES of the task starts here***
[FEW-SHOT EXAMPLE]
***EXAMPLES of the task ends here***

Now, given what you learned from the LABEL DESCRIPTIONS and the EXAMPLES above, please evaluate the YOUTUBE VIDEO METADATA on their stance towards the specified MYTH and provide justification on your label with direct, brief EXCERPT(s) from the YOUTUBE VIDEO METADATA. ONLY EXTRACT INTENTIONAL SHORT, BRIEF EXCERPTS TO JUSTIFY YOUR LABEL. Remember to assign a label based on the video’s stance: label it as "Supports the OUD myth (1)" if it supports or promotes the MYTH, including subtle undertones and implications; "Opposes the OUD myth (-1)" if it debunks or opposes the MYTH; and "Neither (0)" if it is neither supporting nor opposing the specified MYTH.

FORMAT your response as a JSON object in the following structure [(LABEL, EXCERPTS, JUSTIFICATION)]. Make sure to have the keys LABEL, EXCERPTS, JUSTIFICATION in the JSON structure. Be concise and clear.

****YOUTUBE VIDEO METADATA to be evaluated starts here ****:

- Video Title: [TITLE]
- Video Description: [DESCRIPTION]
- Video Transcript: [TRANSCRIPT]
- Video Tags: [TAGS]
   
****YOUTUBE VIDEO METADATA to be evaluated ends here ****.
"""

### M1 FEW SHOT EXAMPLES ###
# Example 1 (Video ID): SjCZwqEE22Y
# Example 2 (Video ID): 7PT0gv6a97o
# Example 3 (Video ID): X3UKcHR-2uU
# Example 4 (Video ID): bMitni3tC-c
# Example 5 (Video ID): fTcGyWDDg5s
M1_FEW_SHOT_EXAMPLES = """
EXAMPLE 1 starts here ****:
VIDEO_TITLE: ""
VIDEO_DESCRIPTION: ""
VIDEO_TRANSCRIPT: ""
VIDEO_TAGS: ''
LABEL: "Opposes the OUD myth (-1)"
REASONING: The video description states that "Suboxone is a medication used to help people overcome addiction to opioids," explaining what suboxone is and acknowledging their efficacy to treat opioid use disorder. By acknowledging the efficacy of MAT and their usage, this video metadata opposes the myth by clarifying that medication-assisted treatment is actually for treating opioid use disorder, rather than replacing one drug with another. 

EXAMPLE 2 starts here ****:
VIDEO_TITLE: ""
VIDEO_DESCRIPTION: ""
VIDEO_TRANSCRIPT: ""
VIDEO_TAGS: ""
LABEL: "Neither (0)"
REASONING: The video discusses Kratom and their effects in regards to other opioids; however, Kratom is not a MAT and thus do not relate to the myth at all. Thus, videos solely discussing Kratom should be marked as "Neither (0)" as they are not applicable to the myth.

EXAMPLE 3 starts here ****:
VIDEO_TITLE: ""
VIDEO_DESCRIPTION: ""
VIDEO_TRANSCRIPT: ""
VIDEO_TAGS: ''
LABEL: "Opposes the OUD myth (-1)"
REASONING: The video transcript discusses babies born addicted to heroin who need methadone to wean them off of heroin. This is factual and actually promotes the usage of methadone as a medication to treat opioid addiction, thus opposing the myth. Note that videos that promote the usage of methadone and/or acknowledges their efficacy for treatment as a medication should be marked as opposing the OUD myth. Note that methadone is a form of opioid used for medication to treat opioid use disorder; in other words, you need opioids like methadone to treat opioid addiction on drugs like fentanyl and heroin.

EXAMPLE 4 starts here ****:
VIDEO_TITLE: ""
VIDEO_DESCRIPTION: ""
VIDEO_TRANSCRIPT: ""
VIDEO_TAGS: ''
LABEL: "Supports the OUD myth (1)"
REASONING: The video transcript says "if you're looking at helping a friend through methadone detox odds are your friend has a substantial drug problem to begin with." This implies that using methadone is equivalent to having a "substantial drug problem," indicating that using such medication is similar to having a drug issue. Thus, this supports the myth. Medication assisted treatment can be life-changing and positive for people with OUD, and tapering down from MAT can be suitable for some people (but not all) under proper doctor supervision. Any videos that likens methadone and other MAT to a drug issue or addiction should be marked as "Supporting the OUD myth," including subtle tones like calling people who got off MAT as being "clean as a whistle" and encouraging people to get off of MAT "with no opioid support."
 
EXAMPLE 5 starts here ****:
VIDEO_TITLE: ""
VIDEO_DESCRIPTION: ""
VIDEO_TRANSCRIPT: ""
VIDEO_TAGS: ''
LABEL: "Supports the OUD myth (1)"
REASONING: Although the video discusses suboxone positively as a treatment medication, the transcript calls methadone a "pure replacement" to drugs like heroin and oxycontin. This negative framing supports the OUD myth. Other videos that negatively frames methadone as one that "dumbs you down as a person," "gives you a buzz from it," "one needs to be done with," and/or "liquid handcuffs" should be marked as supporting the OUD myth, as it likens the medication as illicit drugs like heroin and fentanyl.
"""

### M2 FEW SHOT EXAMPLES ###
# Example 1 (Video ID): 9TYr6sqDEfY
# Example 2 (Video ID): DyjRxf-aJN4
# Example 3 (Video ID): AnUN2Zs4Mnk
# Example 4 (Video ID): m_uV8UkTDKw
# Example 5 (Video ID): -3G162dqVVI
M2_FEW_SHOT_EXAMPLES = """EXAMPLE 1 starts here ****:
VIDEO_TITLE: ""
VIDEO_DESCRIPTION: ""
VIDEO_TRANSCRIPT: ""
VIDEO_TAGS: ''
LABEL: "Supports the OUD myth (1)"
REASONING: The video transcript contains the excerpt “you become a heroin monster which basically means that you will go and Rob your grandmother,” suggesting that people with OUD will rob and steal from others to maintain their addiction to heroin. Thus, the video supports the myth that people with OUD have self-imposed conditions maintained by the lack of moral fiber.

EXAMPLE 2 starts here ****:
VIDEO_TITLE: ""
VIDEO_DESCRIPTION: ""
VIDEO_TRANSCRIPT: ""
VIDEO_TAGS: ''
LABEL: "Supports the OUD myth (1)"
REASONING: The video transcript contains the excerpt "it's uh up to you it's you know there's a there's ways to do it yeah you just got to want to do it," which emphasizes the individual's willpower and moral responsibility to fight OUD. The individual's "want to do it" aligns with the stigmatizing belief that OUD results from a lack of moral strength or personal determination rather than acknowledging it as a complex medical condition requiring evidence-based treatment and medication. Thus, the video supports the myth. Videos urging people to adopt the "mentality" to quit illicit opioids or accusing them of wanting to "stay in their addiction" should be marked as supporting, as they frame individuals with OUD as lacking willpower or moral responsibility.

EXAMPLE 3 starts here ****:
VIDEO_TITLE: ""
VIDEO_DESCRIPTION: ""
VIDEO_TRANSCRIPT: ""
VIDEO_TAGS: ''
LABEL: "Neither (0)"
REASONING: The video discusses Kratom and the history of the opioid epidemic. Despite discussing their addictive nature and the societal impacts of the opioid epidemic, the video should be labeled as "Neither (0)" given that Kratom is not an FDA-approved medication to treat OUD. As demonstrated by this example, any videos that solely discuss Kratom should be marked as "Neither (0)."

EXAMPLE 4 starts here ****:
VIDEO_TITLE: ""
VIDEO_DESCRIPTION: ""
VIDEO_TRANSCRIPT: ""
VIDEO_TAGS: ''
LABEL: "Supports the OUD myth (1)"
REASONING: This video transcript contains the excerpt "I think that if you do it the way that you're supposed and you don't cheat and you don't play around with them and you don't take more than you should and you know come in and lie about it then you're gonna be okay," framing recovery from OUD primarily as a matter of adhering to moral and behavioral standards. Although the video discusses the benefits of suboxone for the person, the emphasis on “not cheating the program” and “not lying about it” implies that success in treatment is primarily a matter of honesty and moral behavior, which aligns with the stigmatizing belief that OUD is a result of personal failings rather than a chronic medical condition. By focusing on behavior rather than the medical aspects of OUD, the video supports the myth.

EXAMPLE 5 starts here ****:
VIDEO_TITLE: ""
VIDEO_DESCRIPTION: ""
VIDEO_TRANSCRIPT: ""
VIDEO_TAGS: ''
LABEL: "Supports the OUD myth (1)"
REASONING: The video transcript contains the excerpt "WELL, IF I’M THE DRUG DEALER I’M GOING TO STAND AT THE METHADONE CLINIC AND THAT’S WHAT’S HAPPENIN," which subtly implies that people with OUD who attends methadone clinics lack moral fiber and will buy from drug dealers standing at the methadone clinic. Although the video contains counterperspectives supporting methadone clinics for people with OUD, the video excerpt frames OUD not as a medical disease, but more of a criminal issue that attracts drug dealers and makes the neighborhood unsafer.
"""

### M3 FEW SHOT EXAMPLES ###
# Example 1 (Video ID): Jc-buPCKisM
# Example 2 (Video ID): 0hR2Hwkhey8
# Example 3 (Video ID): UfQWOGOFNFA
# Example 4 (Video ID): JczoO7ogOS8
# Example 5 (Video ID): SjCZwqEE22Y
M3_FEW_SHOT_EXAMPLES = """EXAMPLE 1 starts here ****:
VIDEO_TITLE: ""
VIDEO_DESCRIPTION: ""
VIDEO_TRANSCRIPT: ""
VIDEO_TAGS: ''
LABEL: "Opposes the OUD myth (-1)"
REASONING: The video transcript contains the excerpt "Suboxone buprenorphine it's a partial opioid Agonist which means it partially turns on that opioid receptor not a true opioid but it's a treatment for opioid use disorder cuts the death rate in half totally kills cravings and withdrawals and makes people feel quote normal," dicussing what suboxone is and acknowledging their effectiveness in treating and helping people recover. By explaining the effectiveness of MAT for helping people recover, the video opposes the myth that taking medication is not true recovery.

EXAMPLE 2 starts here ****:
VIDEO_TITLE: ""
VIDEO_DESCRIPTION: ""
VIDEO_TRANSCRIPT: ""
VIDEO_TAGS: ''
LABEL: "Supports the OUD myth (1)"
REASONING: The video frames medication opioids like suboxone and methadone as "addiction" and promotes methods, such as "sleep through withdrawals," rather than relying on MAT to treat people with OUD. Such videos that promote abstinence-based methods, such as cold-turkeying, advanced rapid detox, and sleeping through withdrawals should be marked as supporting the OUD myth.

EXAMPLE 3 starts here ****:
VIDEO_TITLE: ""
VIDEO_DESCRIPTION: ""
VIDEO_TRANSCRIPT: ""
VIDEO_TAGS: ''
LABEL: "Opposes the OUD myth (-1)"
REASONING: Although the video promotes suboxone tapering, the video description says "But remember, not everyone Is ready to taper.  And If you’re not ready, that’s okay.", which suggests that long-term maintenance on medication-assisted treatment is valid. The video avoids stigmatizing long-term usage of MAT and encourages a potential way to taper down from suboxone for patients who are looking to do so under doctor supervision, which is valid too. Thus, the video opposes the myth by acknowledging the long-term maintenance aspect of MAT and suggesting a way to taper from MAT. Note that videos that stigmatize long-term usage of MAT, urges people to get off medication such as suboxone and methadone, and/or presents tapering as the only solution should be labeled as "Supports the OUD myth (1)." However, this video does acknowledge that MAT maintenance is a valid option -- it could be a patient's goal to taper off MAT under doctor supervision.

EXAMPLE 4 starts here ****:
VIDEO_TITLE: ""
VIDEO_DESCRIPTION: ""
VIDEO_TRANSCRIPT: ""
VIDEO_TAGS: ''
LABEL: "Supports the OUD myth (1)"
REASONING: The video transcript contains the excerpt "I don't want to take suboxone long term," which suggests that suboxone is something negative and one should get off of as soon as possible. This supports the myth that the ultimate goal of treatment is abstinence from any opioid use, particularly medication opioids such as suboxone.

EXAMPLE 5 starts here ****:
VIDEO_TITLE: ""
VIDEO_DESCRIPTION: ""
VIDEO_TRANSCRIPT: ""
VIDEO_TAGS: ''
LABEL: "Opposes the OUD myth (-1)"
REASONING: The video transcript contains the excerpt "if you're watching this chances are you've taken the critical step to overcome opioid addiction and are looking towards a bright future in recovery," acknowledging. the effectiveness of suboxone in treatment for OUD and recovery. In addition, the video also argues against cold-turkeying and promotes tapering under doctor supervision (e.g., "why tapering is recommended over quitting cold turkey the difference between short and long tapers and steps on getting off Suboxone tapering off Suboxone is considered the safest way to discontinue medication used under the supervision of a doctor"), which opposes the myth. Given these contexts of acknowledging the role of suboxone in treatment for OUD and recovery and denouncing cold-turkeying (e.g., a disproven method using abstinence), the video opposes the myth.
"""

### M4 FEW SHOT EXAMPLES ###
# Example 1 (Video ID): DyjRxf-aJN4
# Example 2 (Video ID): 7PT0gv6a97o
# Example 3 (Video ID): Qg7G0VTi3iY
# Example 4 (Video ID): OFGFeA6Ap7E
# Example 5 (Video ID): v4GnSSvcYys
M4_FEW_SHOT_EXAMPLES = """EXAMPLE 1 starts here ****:
VIDEO_TITLE: ""
VIDEO_DESCRIPTION: ""
VIDEO_TRANSCRIPT: ""
VIDEO_TAGS: ''
LABEL: "Supports the OUD myth (1)"
REASONING: The video transcript contains the excerpt "it's uh up to you it's you know there's a there's ways to do it yeah you just got to want to do it," which suggests that patients who lack moral strength and personal determination are vulnerable to addiction and thus may not recover from OUD. Thus, the video supports the myth. Videos urging people to adopt the "mentality" to quit illicit opioids or accusing them of wanting to "stay in their addiction" should be marked as supporting, as they subtly imply that individuals who lack willpower or moral responsibility are vulnerable to addiction and will continue having OUD.

EXAMPLE 2 starts here ****:
VIDEO_TITLE: ""
VIDEO_DESCRIPTION: ""
VIDEO_TRANSCRIPT: ""
VIDEO_TAGS: ''
LABEL: "Neither (0)"
REASONING: The video itself simply discusses Kratom as a substance that is not well understood, thus not being relevant to the myth that people with certain characteristics are vulnerable to addiction.

EXAMPLE 3 starts here ****:
VIDEO_TITLE: ""
VIDEO_DESCRIPTION: ""
VIDEO_TRANSCRIPT: ""
VIDEO_TAGS: ''
LABEL: "Opposes the OUD myth (-1)"
REASONING: The video states that anyone from various economic backgrounds and different races and colors is being affected by the deadly drug (e.g., fentanyl). This directly opposes the myth by suggesting that anyone is vulnerable to addiction, rather than patients with certain characteristics. Please make sure not to hallucinate this example into the provided video metadata below.

EXAMPLE 4 starts here ****:
VIDEO_TITLE: ""
VIDEO_DESCRIPTION: ""
VIDEO_TRANSCRIPT: ""
VIDEO_TAGS: ''
LABEL: "Neither (0)"
REASONING: The video discusses the effectiveness of buprenorphine in helping people manage opioid addiction. However, the video does not imply anything about how people with certain characteristics are vulnerable to ADDICTION. Thus, the video is labeled as "Neither (0)."

EXAMPLE 5 starts here ****:
VIDEO_TITLE: ""
VIDEO_DESCRIPTION: ""
VIDEO_TRANSCRIPT: ""
VIDEO_TAGS: ''
LABEL: "Neither (0)"
REASONING: The video highlights Jonel's background and circumstances, such as living on reservations and experiencing family instability. However, the video does not address the myth that patients with certain CHARACTERISTICS are vulnerable to addiction.
"""

### M5 FEW SHOT EXAMPLES ###
# Example 1 (Video ID): zN9NDZ6lgaM
# Example 2 (Video ID): 7PT0gv6a97o
# Example 3 (Video ID): sZ5-i72Yl2Q
# Example 4 (Video ID): FmGalSsq63k
# Example 5 (Video ID): TnYHKxUHgCs
M5_FEW_SHOT_EXAMPLES = """EXAMPLE 1 starts here ****:
VIDEO_TITLE: ""
VIDEO_DESCRIPTION: ""
VIDEO_TRANSCRIPT: ""
VIDEO_TAGS: ''
LABEL: "Supports the OUD myth (1)"
REASONING: The video transcript says "methadone is available on the black market in both pill and liquid form and is as addictive as the substances it replaces," which perpetuates the myth that physical dependence or tolerance is the same as addiction. Methadone helps individuals regain stability in their lives by managing their physical withdrawal symptoms and reducing cravings. Although methadone can cause physical dependence or tolerance, the video argues that methadone is as addictive as the drug it replaces, which perpetuates the myth that physical dependence is the same as addiction.

EXAMPLE 2 starts here ****:
VIDEO_TITLE: ""
VIDEO_DESCRIPTION: ""
VIDEO_TRANSCRIPT: ""
VIDEO_TAGS: ''
LABEL: "Neither (0)"
REASONING: The video discusses whether Kratom is similar to heroin and fentanyl, which does not relate to the myth that physical dependence or tolerance is the same as addiction. Thus, the video is labeled as "Neither (0)."

EXAMPLE 3 starts here ****:
VIDEO_TITLE: ""
VIDEO_DESCRIPTION: ""
VIDEO_TRANSCRIPT: ""
VIDEO_TAGS: ''
LABEL: "Supports the OUD myth (1)"
REASONING: The video transcript contains the excerpt that "I would do [suboxone] for a couple of like maybe a week don't do it long because yes you're gonna get addicted to suboxone," suggesting that suboxone will get people high and addicted. Thus, the video supports the myth by conflating that physical dependence or tolerance from suboxone is the same as addiction. 

EXAMPLE 4 starts here ****:
VIDEO_TITLE: ""
VIDEO_DESCRIPTION: ""
VIDEO_TRANSCRIPT: ""
VIDEO_TAGS: ''
LABEL: "Opposes the OUD myth (-1)"
REASONING: The video explains the science background behind opioid addiction and how suboxone works biologically in the brain. By clarifying what addiction is along with how tolerance and dependence forms, the video opposes the myth that physical dependence or tolerance is the same as addiction.

EXAMPLE 5 starts here ****:
VIDEO_TITLE: ""
VIDEO_DESCRIPTION: ""
VIDEO_TRANSCRIPT: ""
VIDEO_TAGS: ''
LABEL: "Opposes the OUD myth (-1)"
REASONING: The video debunks the idea that suboxone makes people high, distinguishing between physical dependence and addiction. In particular, the video explains that taking suboxone long-term does not meet the criteria for addiction, thus opposing the myth that physical dependence or tolerance is the same as addiction.
"""        

### M6 FEW SHOT EXAMPLES ###
# Example 1 (Video ID): tzHKfZyevXo
# Example 2 (Video ID): eWdCJm9q1bw
# Example 3 (Video ID): 7gtWuoWGQWM
# Example 4 (Video ID): WNXieqey_iQ
# Example 5 (Video ID): SjCZwqEE22Y
M6_FEW_SHOT_EXAMPLES = """EXAMPLE 1 starts here ****:
VIDEO_TITLE: ""
VIDEO_DESCRIPTION: ""
VIDEO_TRANSCRIPT: ""
VIDEO_TAGS: ''
LABEL: "Supports the OUD myth (1)"
REASONING: The video states that people just need to "put in the effort to taper" and avoid taking suboxone for getting off kratom. In general, videos that promote cold turkey from OUD, oppose any long-term maintenance with medication-assisted treatment (or oppose such medication like suboxone and methadone), and/or solely focus on detox and taper from MAT without medical supervision or without acknowledging that detox and taper is not a solution for everyone (e.g., some people may need to be on long-term maintenance with MAT for life) should be labeled as "Supports the OUD myth (1)." Thus, this video should be marked as supporting the OUD myth given the focus on tapering (without medical supervision) and opposing the usage of MAT, such as suboxone, for long-term maintenance in a person's recovery journey from a kratom addiction. Videos that promote advancedrapiddetox along with detox methods, such as "sleeping through withdrawals," should also be labeled as supporting the OUD myth. 

EXAMPLE 2 starts here ****:
VIDEO_TITLE: ""
VIDEO_DESCRIPTION: ""
VIDEO_TRANSCRIPT: ""
VIDEO_TAGS: ''
LABEL: "Neither (0)"
REASONING: The video discusses a story of a police whose daughter overdosed on heroin. The police realized that the war on drugs and criminalization of drugs were ineffective, and he is planning on creating a recovery house for patients with OUD. This video does not relate to the myth that "Detoxification for OUD is effective," and thus should be labeled as "Neither (0)." 

EXAMPLE 3 starts here ****:
VIDEO_TITLE: ""
VIDEO_DESCRIPTION: ""
VIDEO_TRANSCRIPT: ""
VIDEO_TAGS: ''
LABEL: "Supports the OUD myth (1)"
REASONING: The video promotes cold turkey as an approach to get off of methadone, in particular "you would stop the methadone completely and you' get comfort meds." In addition, the video does not acknowledge that tapering from MAT may not be a solution for everyone. Thus, the video should be marked as "Supports the OUD myth (1)."

EXAMPLE 4 starts here ****:
VIDEO_TITLE: ""
VIDEO_DESCRIPTION: ""
VIDEO_TRANSCRIPT: ""
VIDEO_TAGS: ''
LABEL: "Opposes the OUD myth (-1)"
REASONING: The video discusses the history of suboxone and their role in treating patients with OUD. By discussing MAT and their efficacy in maintenance/treatment for OUD, the video opposes the OUD myth that detoxification is effective. In particular, videos that promote MAT and a structured approach to recovery from OUD (rather than a short-term detox) should be marked as opposing the OUD myth.

EXAMPLE 5 starts here ****:
VIDEO_TITLE: ""
VIDEO_DESCRIPTION: ""
VIDEO_TRANSCRIPT: ""
VIDEO_TAGS: ''
LABEL: "Opposes the OUD myth (-1)"
REASONING: The video explains what suboxone is and their role in helping people overcome their addiction to opioids. In particular, the video recommends people to obtain medical supervision to taper off suboxone safely and effectively, compared to quitting suboxone "cold turkey." Thus, the video should be marked as opposing the OUD myth as it debunks detoxification methods such as "cold turkey" and recommends a MAT tapering approach backed by medical supervision.
"""

### M7 FEW SHOT EXAMPLES ###
# Example 1 (Video ID): QtRQ9UD7rpY
# Example 2 (Video ID): W-7_alg4I28
# Example 3 (Video ID): 0RkpSTlvvj0
# Example 4 (Video ID): 0y55ymuJ2K4
# Example 5 (Video ID): TP0ToVYXQ-k
M7_FEW_SHOT_EXAMPLES = """EXAMPLE 1 starts here ****:
VIDEO_TITLE: ""
VIDEO_DESCRIPTION: ""
VIDEO_TRANSCRIPT: ""
VIDEO_TAGS: ''
LABEL: "Neither (0)"
REASONING: The video promotes kratom as a potential safe alternative to harder drugs, such as heroin, and advises people to start "weaning" off of Kratom. However, kratom is not an approved medication for treating OUD, thus the video is not relevant to the myth that "you should only take medication for a brief period of time." As such, the video is labeled "Neither (0)".

EXAMPLE 2 starts here ****:
VIDEO_TITLE: ""
VIDEO_DESCRIPTION: ""
VIDEO_TRANSCRIPT: ""
VIDEO_TAGS: ''
LABEL: "Neither (0)"
REASONING: The video offers advice on suboxone withdrawal, recommending finding a qualified doctor who can help plan a slow, structured suboxone taper schedule. Given that the video is recommending a medically-supervised approach to taper (which for some patient is a goal to get off of suboxone taper after recovering from OUD), the video is labeled as "Neither (0)" as it does not relate to the myth that the medication should only be taken briefly.

EXAMPLE 3 starts here ****:
VIDEO_TITLE: ""
VIDEO_DESCRIPTION: ""
VIDEO_TRANSCRIPT: ""
VIDEO_TAGS: ''
LABEL: "Supports the OUD myth (1)"
REASONING: The video states "people [be on] suboxone and methadone for years," suggesting that patients with OUD should not be on medications like suboxone and methadone for a long period of time. Thus, the video perpetuates the myth that you should only take medication for a brief period of time. Any video that has the sentiment that one should not be on methadone or suboxone for too long should be marked as "Supports the OUD myth (1)."

EXAMPLE 4 starts here ****:
VIDEO_TITLE: ""
VIDEO_DESCRIPTION: ""
VIDEO_TRANSCRIPT: ""
VIDEO_TAGS: ''
LABEL: "Supports the OUD myth (1)"
REASONING: The video focuses on a person's "detox" journey from methadone and oxycontin using ibogaine. Although it could be a person's goal to taper off of methadone and suboxone after recovering from OUD, the video places a strong emphasis on "taper" and "detox" (e.g., getting off methadone), which supports the myth that one should only take medication for a brief period of time. Similarly, videos that promote Advanced Rapid Detox (e.g., advancedrapiddetox) and place huge emphasis on getting off MAT without considering the benefits of long-term maintenance of MAT should be marked as "Supports the OUD myth (1)."

EXAMPLE 5 starts here ****:
VIDEO_TITLE: ""
VIDEO_DESCRIPTION: ""
VIDEO_TRANSCRIPT: ""
VIDEO_TAGS: ''
LABEL: "Opposes the OUD myth (-1)"
REASONING: The video explains how a person can start a suboxone taper. The doctor emphasizes that treatment plans are individualized, stating that "everybody's different people are on a different plan not everybody needs to be tapering people need you know you need to be stable right and you need to talk to your doctor and find out what's right for you." The video acknowledges the benefits of long-term MAT maintenance and recommends talking to one's doctor to find a proper tapering plan. Unlike prior examples discussing MAT taper (e.g., EXAMPLE 4), the video balances tapering advice with the acknowledgment that medication should be taken as long as needed, countering the myth that MAT is only for short-term use.
"""

### M8 FEW SHOT EXAMPLES ###
# Example 1 (Video ID): 6F6d10ggVDw
# Example 2 (Video ID): nmMCQ1y8l14
# Example 3 (Video ID): GI3blNNe56w
# Example 4 (Video ID): E9jKyHjPbUg
# Example 5 (Video ID): QtRQ9UD7rpY
M8_FEW_SHOT_EXAMPLES = """EXAMPLE 1 starts here ****:
VIDEO_TITLE: ""
VIDEO_DESCRIPTION: ""
VIDEO_TRANSCRIPT: ""
VIDEO_TAGS: ''
LABEL: "Opposes the OUD myth (-1)"
REASONING: The video discusses the potential dangers of taking Kratom, such as addiction, seizuire, and even death. This directly counters the myth that Kratom is a non-addictive and safe alternative to opioids. Thus, the video is labeled as "Opposes the OUD myth (-1)."

EXAMPLE 2 starts here ****:
VIDEO_TITLE: ""
VIDEO_DESCRIPTION: ""
VIDEO_TRANSCRIPT: ""
VIDEO_TAGS: ''
LABEL: "Supports the OUD myth (1)"
REASONING: The video discusses a person's experience taking Kratom, arguing that "I know it's [addiction] swapping but if you switch to [Kratom] and you take [Kratom] for the next ten years that is so much better than doing heroin" and "[Kratom] is so much better for people it's so much safer it's healthier." Although the video does admit that Kratom is addictive, it also heavily states that it's "safer" and "healthier," which heavily supports the myth that Kratom is a safer alternative to opioids. 

EXAMPLE 3 starts here ****:
VIDEO_TITLE: ""
VIDEO_DESCRIPTION: ""
VIDEO_TRANSCRIPT: ""
VIDEO_TAGS: ''
LABEL: "Neither (0)"
REASONING: The video focuses on preparing kratom tea, rather than discussing its safety or addictive aspects. Given that the video is neutral about Kratom, the video is labeled as "Neither (0)."

EXAMPLE 4 starts here ****:
VIDEO_TITLE: ""
VIDEO_DESCRIPTION: ""
VIDEO_TRANSCRIPT: ""
VIDEO_TAGS: ''
LABEL: "Supports the OUD myth (1)"
REASONING: The video states "[Kratom] also can be used to actually help withdrawals for opiates," which implies that Kratom is a helpful and safe alternative to opioids. Thus, the video supports the myth that Kratom is a non-addictive and safe alternative to opioids.

EXAMPLE 5 starts here ****:
VIDEO_TITLE: ""
VIDEO_DESCRIPTION: ""
VIDEO_TRANSCRIPT: ""
VIDEO_TAGS: ''
LABEL: "Supports the OUD myth (1)"
REASONING: The video promotes kratom as a potential safe alternative to harder drugs, such as heroin, and advises people to start "weaning" off of Kratom. As such, the video is labeled "Supports the OUD myth (1)."
"""
