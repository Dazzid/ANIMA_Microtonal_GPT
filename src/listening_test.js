/*const success_url = 'https://app.prolific.co/submissions/complete?cc=C12KY3JS';
*/
var timeline = [];

function addStyles(styles) {
    var styleSheet = document.createElement("style");
    styleSheet.type = "text/css";
    styleSheet.innerText = styles;
    document.head.appendChild(styleSheet);
}

var opt_scale = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'];
var gender = ['Male', 'Female', 'Non-binary', 'Prefer not to say'];

var context = [
  "CvMvM7[I] G#vMvM7/C[VI] C#vMvM7[II] F^m^m7[IV] D#vMm7[III] CvMvM7[I]",
  "CNN7[I] CNN7/vvE[I] ^^Fø^m7[IV] vvAnN7[VI] ^Døvm7[II] ^Gn^m7[V] CNN7[I]",
  "CvMvM7[I] C#vMvM7/F[II] F#vMvM7[V] Bb^m^m7[VII] G#vMm7[VI] CvMvM7[I]",
  "CNN7[I] vvENN7/^G[III] vvAnN7[VI] CNN7[I] ^^Fø^m7[IV] vvBNm7[VII] CNN7[I]",
  "CvMvm7[I] C^m^m7/D#[I] ^Fvm7[IV] vG#maj7[VI] Dø7[II] G^m^m7[V] CvMvm7[I]",
  "C^m^m7[I] vD^m^m7/F[II] vGvM^m7[V] vvBø^m7[VII] D#[III] vAø7[VI] C^m^m7[I]",
  "CvMvm7[I] ^DvMvm7/F#[II] ^GvMvM7[V] B^m^m7[VII] E^m^m7[III] ^Avm7[VI] CvMvm7[I]",
  "C^m^m7[I] CvM^m7/vE[I] vvF#ø^m7[IV] vA^m^m7[VI] vDvM^m7[II] G^mN7[V] C^m^m7[I]",
  "CvMvm7[I] G#vMvm7/C[VI] C#vMvM7[II] F^m^m7[IV] Bb^m^m7[VII] D#vm7[III] CvMvm7[I]",
  "CvMvM7[I] D#vMvM7/G[III] G#vMvM7[VI] C^m^m7[I] ^BbvMm7[VII] CvMvM7[I]"
];

var all_audio_samples = [
  "prog_001.mp3",
  "prog_002.mp3",
  "prog_003.mp3",
  "prog_004.mp3",
  "prog_005.mp3",
  "prog_006.mp3",
  "prog_007.mp3",
  "prog_008.mp3",
  "prog_009.mp3",
  "prog_010.mp3"
];

var question_simple = [
  "0001_I_IV_V_vi.mp3",
  "0002_ii_V_I_vi.mp3",
  "0003_vi_ii_V_I.mp3"
  ];


/* -------------------------------------------------------
   CHORD DEFINITIONS for Part 1 (20 types x 3 files each)
   ------------------------------------------------------- */
var chordTypes = [
  { num: "01", cat: "Consonant",      name: "vMvM7",  files: [
    "01_Consonant_A3_vMvM7.wav", "01_Consonant_Bb3_vMvM7.wav", "01_Consonant_A%233_vMvM7.wav"] },
  { num: "02", cat: "Consonant",      name: "vM^m7",  files: [
    "02_Consonant_B3_vM^m7.wav", "02_Consonant_^^B3_vM^m7.wav", "02_Consonant_vC4_vM^m7.wav"] },
  { num: "03", cat: "Consonant",      name: "^m^m7",  files: [
    "03_Consonant_^A3_^m^m7.wav", "03_Consonant_vB3_^m^m7.wav", "03_Consonant_^B3_^m^m7.wav"] },
  { num: "04", cat: "Intermediate",   name: "SMSM7",  files: [
    "04_Intermediate_^A3_SMSM7.wav", "04_Intermediate_B3_SMSM7.wav", "04_Intermediate_^^B3_SMSM7.wav"] },
  { num: "05", cat: "Intermediate",   name: "SMm7",   files: [
    "05_Intermediate_^A%233_SMm7.wav", "05_Intermediate_vvB3_SMm7.wav", "05_Intermediate_B3_SMm7.wav"] },
  { num: "06", cat: "Intermediate",   name: "maj7",   files: [
    "06_Intermediate_vvB3_maj7.wav", "06_Intermediate_vB3_maj7.wav", "06_Intermediate_B3_maj7.wav"] },
  { num: "07", cat: "Intermediate",   name: "m7",     files: [
    "07_Intermediate_^A3_m7.wav", "07_Intermediate_vBb3_m7.wav", "07_Intermediate_vC4_m7.wav"] },
  { num: "08", cat: "Neutral",        name: "Mn7",    files: [
    "08_Neutral_^A3_Mn7.wav", "08_Neutral_^A%233_Mn7.wav", "08_Neutral_B3_Mn7.wav"] },
  { num: "09", cat: "Neutral",        name: "Nn7",    files: [
    "09_Neutral_A3_Nn7.wav", "09_Neutral_vvB3_Nn7.wav", "09_Neutral_B3_Nn7.wav"] },
  { num: "10", cat: "Neutral",        name: "nn7",    files: [
    "10_Neutral_^^A3_nn7.wav", "10_Neutral_Bb3_nn7.wav", "10_Neutral_^B3_nn7.wav"] },
  { num: "11", cat: "Less_consonant", name: "N^M7",   files: [
    "11_Less_consonant_vBb3_N^M7.wav", "11_Less_consonant_A%233_N^M7.wav", "11_Less_consonant_^A%233_N^M7.wav"] },
  { num: "12", cat: "Less_consonant", name: "vMsm7",  files: [
    "12_Less_consonant_^^A3_vMsm7.wav", "12_Less_consonant_^B3_vMsm7.wav", "12_Less_consonant_vC4_vMsm7.wav"] },
  { num: "13", cat: "Less_consonant", name: "Msm7",   files: [
    "13_Less_consonant_^A%233_Msm7.wav", "13_Less_consonant_vvB3_Msm7.wav", "13_Less_consonant_vB3_Msm7.wav"] },
  { num: "14", cat: "Less_consonant", name: "smsm7",  files: [
    "14_Less_consonant_A%233_smsm7.wav", "14_Less_consonant_^B3_smsm7.wav", "14_Less_consonant_^^B3_smsm7.wav"] },
  { num: "15", cat: "Dominant",       name: "M^m7",   files: [
    "15_Dominant_Bb3_M^m7.wav", "15_Dominant_A%233_M^m7.wav", "15_Dominant_^^B3_M^m7.wav"] },
  { num: "16", cat: "Dominant",       name: "vM^m7",  files: [
    "16_Dominant_Bb3_vM^m7.wav", "16_Dominant_A%233_vM^m7.wav", "16_Dominant_vB3_vM^m7.wav"] },
  { num: "17", cat: "Dominant",       name: "vMvm7",  files: [
    "17_Dominant_^A3_vMvm7.wav", "17_Dominant_^^A3_vMvm7.wav", "17_Dominant_B3_vMvm7.wav"] },
  { num: "18", cat: "Diminished",     name: "ø^m7",   files: [
    "18_Diminished_A%233_ø^m7.wav", "18_Diminished_^A%233_ø^m7.wav", "18_Diminished_^^B3_ø^m7.wav"] },
  { num: "19", cat: "Diminished",     name: "søm7",   files: [
    "19_Diminished_^A3_søm7.wav", "19_Diminished_^^A3_søm7.wav", "19_Diminished_B3_søm7.wav"] },
  { num: "20", cat: "Diminished",     name: "ø7",     files: [
    "20_Diminished_A3_ø7.wav", "20_Diminished_Bb3_ø7.wav", "20_Diminished_vvB3_ø7.wav"] }
];

/* Reference chords for calibration (not rated) */
var referenceChords = {
  dissonant: { num: "21", cat: "Reference", name: "Mpsm7", files: [
    "21_Reference_Bb3_Mpsm7.wav", "21_Reference_^A%233_Mpsm7.wav", "21_Reference_vB3_Mpsm7.wav"] },
  consonant: { num: "22", cat: "Reference", name: "maj6",  files: [
    "22_Reference_A3_maj6.wav", "22_Reference_vB3_maj6.wav", "22_Reference_^B3_maj6.wav"] }
};

/* Build all 60 chord filenames for preloading */
var allChordFiles = [];
for (var ci = 0; ci < chordTypes.length; ci++) {
  for (var fi = 0; fi < chordTypes[ci].files.length; fi++) {
    allChordFiles.push(chordTypes[ci].files[fi]);
  }
}

/* Add reference chord files to preload list */
var allReferenceFiles = [].concat(referenceChords.dissonant.files, referenceChords.consonant.files);


/* -------------------------------------------------------
   PRELOAD (all audio: progressions + screening + chords + references + noise)
   ------------------------------------------------------- */
var allPreloadAudio = [].concat(all_audio_samples, question_simple, allChordFiles, allReferenceFiles, ["00_white_noise_clean.wav"]);
console.log("=== FILES TO PRELOAD (" + allPreloadAudio.length + ") ===");
console.log(allPreloadAudio);

var preload = {
  type: jsPsychPreload,
  audio: allPreloadAudio,
  continue_after_error: true,
  on_finish: function(data) {
    console.log("=== PRELOAD RESULTS ===");
    console.log("Success:", data.success);
    console.log("Failed audio:", data.failed_audio);
  }
};

var timeline = [preload];

const fontStyles = `
body, .jspsych-display-element, .jspsych-content {
  font-family: 'Fira Code', 'Monaco', 'SF Mono', 'Consolas', monospace;
}
`;

/* Load Fira Code from Google Fonts */
var fontLink = document.createElement("link");
fontLink.href = "https://fonts.googleapis.com/css2?family=Fira+Code:wght@300;400;500;600;700&display=swap";
fontLink.rel = "stylesheet";
document.head.appendChild(fontLink);

const buttonStyles = `
.jspsych-btn {
  background-color: #ffaa00; /* Orange */
  border: none;
  color: white;
  padding: 10px 24px;
  text-align: center;
  text-decoration: none;
  display: inline-block;
  font-size: 16px;
  margin: 4px 2px;
  cursor: pointer;
  border-radius: 12px;
  transition: background 0.3s, transform 0.3s;
}

.jspsych-btn:hover {
  background-color: #ffaa00;
  transform: scale(1.025);
}
`;

const preambleStyles = `
.preamble-text {
  font-size: 24px;
}

.small-text {
  font-size: 16px;
}

.audio-container {
  background-color: #0099FF99;
  display: inline-block;
  margin: 20px 20px 50px;
  padding: 5px;
  border-radius: 15px;
  border-bottom: 2px solid #0003; /* Example: 2px solid black line */
  
}
`;

const questionStyles = `
.jspsych-survey-multi-choice-question:after, .jspsych-survey-text-question:after,
.jspsych-survey-multi-choice-question::after, .jspsych-survey-text-question::after,
.jspsych-survey-multi-choice-question .required,
p.jspsych-survey-multi-choice-question span.required {
  content: none  !important;
  display: none !important;
}
`;

const scaleButtonStyles = `
.jspsych-survey-multi-choice .jspsych-btn, .jspsych-survey-multi-choice-horizontal .jspsych-btn {
  font-size: 25px;
  padding: 5px 10px;
  background-color: #007bff; /* Blue */
  color: #ffffff;
}

.jspsych-survey-multi-choice .jspsych-btn:hover, .jspsych-survey-multi-choice-horizontal .jspsych-btn:hover {
  background-color: #0056b3;
}
`;

const audioStyles = `
/* Add your audio player styles here if needed */
`;


const radialButtonStyles = `
/* Target the containers for each option to apply a horizontal layout */
.jspsych-survey-multi-choice-option {
  display: inline-block; /* Align options horizontally */
  margin-right: 15px; /* Space between options */
}

/* Hide the actual input (radio button) visually but keep it accessible */
.jspsych-survey-multi-choice-option input[type="radio"] {

 display: none;
 
}

.jspsych-survey-multi-choice-option label {
  background-color: #f0f0f0; /* Light background for the label */
  padding: 5px 20px 5px; /* Padding around the text */
  border-radius: 20px; /* Rounded corners for the labels */
  font-size: 18px; /* Adjust font size as needed */
  cursor: pointer; /* Pointer cursor on hover */
  display: inline-block; /* Ensure label is inline-block for proper spacing */
  transition: background-color 0.3s, color 0.3s; /* Smooth transition for color changes */
}

/* Change label background and text color when radio button is selected */
.jspsych-survey-multi-choice-option label:has(> input[type="radio"]:checked) {
  background-color: #ffaa00; /* Orange background for selected option */
  color: white; /* White text for selected option */
}
`;

const part1Styles = `
.trial-counter {
  font-size: 14px;
  color: #666;
  margin-bottom: 6px;
}
.noise-screen {
  font-size: 24px;
  color: #999;
  padding: 40px;
}
`;

const calibrationStyles = `
.calibration-container {
  max-width: 640px;
  margin: 0 auto;
  text-align: left;
}
.calibration-card {
  display: inline-block;
  vertical-align: top;
  width: 280px;
  margin: 10px 15px;
  padding: 20px;
  border-radius: 14px;
  text-align: center;
}
.calibration-card.consonant {
  background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
  border: 2px solid #4CAF50;
}
.calibration-card.dissonant {
  background: linear-gradient(135deg, #fbe9e7, #ffccbc);
  border: 2px solid #F44336;
}
.calibration-card .card-label {
  font-size: 20px;
  font-weight: bold;
  margin-bottom: 6px;
}
.calibration-card .card-score {
  font-size: 14px;
  color: #555;
  margin-bottom: 14px;
}
.calibration-card audio {
  width: 100%;
}
.calibration-check {
  font-size: 22px;
  opacity: 0.3;
  transition: opacity 0.3s;
}
.calibration-check.played {
  opacity: 1.0;
}
`;

// Combine all styles
addStyles(`${fontStyles} ${buttonStyles} ${scaleButtonStyles} ${audioStyles} ${preambleStyles} ${questionStyles} ${radialButtonStyles} ${part1Styles} ${calibrationStyles}`);


var jsPsych = initJsPsych({
    show_progress_bar: true,
    override_safe_mode: true,
    on_finish: function () {
        /* window.location.href = success_url; */
    }
})


// Options for the harmony question
var harmonicOptions = ['ii, V, I, vi', 'I, IV, V, vi', 'vi, ii, V, I', 'iii, IV, V, IV', 'IV, V, VI, ii'];

// Pair each context with its corresponding audio sample
var pairedSamples = context.map((c, i) => ({ context: c, audio: all_audio_samples[i] }));

// Shuffle the paired samples
var shuffledPairs = jsPsych.randomization.shuffle(pairedSamples);

console.log(shuffledPairs)


/* -------------------------------------------------------
   Part 1: Select 1 random file per chord type, shuffle order
   ------------------------------------------------------- */
var selectedChords = [];
for (var sc = 0; sc < chordTypes.length; sc++) {
  var ch = chordTypes[sc];
  var randomFile = jsPsych.randomization.sampleWithoutReplacement(ch.files, 1)[0];
  selectedChords.push({
    num:  ch.num,
    cat:  ch.cat,
    name: ch.name,
    file: randomFile
  });
}
var shuffledChords = jsPsych.randomization.shuffle(selectedChords);
console.log("Part 1 chords:", shuffledChords);

/* Pick one random reference file for each anchor */
var refConsonantFile = jsPsych.randomization.sampleWithoutReplacement(referenceChords.consonant.files, 1)[0];
var refDissonantFile = jsPsych.randomization.sampleWithoutReplacement(referenceChords.dissonant.files, 1)[0];
console.log("Reference consonant:", refConsonantFile);
console.log("Reference dissonant:", refDissonantFile);


/*-------------------------------*/
/*        The Test Starts        */
/*-------------------------------*/

/* 0 Consent Form */
var consentTrial = {
    type: jsPsychHtmlButtonResponse,
    stimulus: `
        <div style="max-width: 740px; margin: 0 auto;">
        <div style="text-align: left; margin-bottom: 20px;">
          <img src="mtg_logo.png" alt="MTG - Music Technology Group" style="max-width: 150px; display: block; margin-bottom: 0;">
        </div>
        <p style="font-size: 25px; background-color: #0afa; border-radius: 20px; padding: 20px; margin-top: 0;">Welcome to the microtonal chords study</p>
        
        <div style="text-align: left; margin-top: 20px;">
        <p style="font-size: 15px; line-height: 1.7;">
          We are <strong>David Dalmazzo</strong> (MTG)<sup>1</sup> and <strong>Ken D&eacute;guernel</strong> (Algomus)<sup>2</sup>, and this is part of the project
          <br><strong>ANIMA</strong> &mdash; <em>Artificial INtelligence-based Interactive Microtonal Compositional Assistant</em>
        </p>
        <p style="font-size: 13px; color: #666; line-height: 1.6;">
          <sup>1</sup> Universitat Pompeu Fabra, Music Technology Group, Barcelona, Spain<br>
          <sup>2</sup> Univ. Lille, CNRS, Centrale Lille, UMR 9189 CRIStAL, F-59000 Lille, France
        </p>

        <hr style="border: none; border-top: 1px solid #ddd; margin: 20px 0;">

        <p style="font-size: 16px;">You have been invited to take part in a listening study that is focused on microtonal chord qualities and progressions.
        <br><strong>The following questionnaire is only compatible with Google Chrome and Safari.</strong>
        <br>The study lasts about 10 minutes. This study is being done in relation to a research project funded by MSCA European Union, Grant Agreement ID: 101203318.
        </p>
        
        <p style="font-size: 20px;">Read the following information carefully before proceeding.</p>
        
          <ul style="font-size: 16px;">
            <li>Your participation is unpaid.</li>
            <li>Your participation is confidential and anonymous. The data you provide are strictly for academic research.</li>
            <li>There are no risks identified in relation to this study.</li>
            <li>This study has been reviewed by a project-specific ethics committee organised by the University Ethical Committee, which has concluded that the personal data collected does not raise ethical concerns.</li>
            <li>If you have questions or concerns after the study, please feel free to contact david.cabrera@upf.edu.</li>
          </ul>

        <hr style="border: none; border-top: 1px solid #ddd; margin: 20px 0;">

        <p style="font-size: 18px; font-weight: bold;">Data Protection Information (GDPR 2016/679)</p>
        <p style="font-size: 14px; line-height: 1.7;">
          <strong>Data controller:</strong> Universitat Pompeu Fabra. C. de la Merc&egrave;, 12. 08002 Barcelona. Tel. +34 93 542 20 00. Data Protection Officer: <a href="mailto:dpd@upf.edu">dpd@upf.edu</a>
        </p>
        <p style="font-size: 14px; line-height: 1.7;">
          <strong>Purposes of the processing:</strong> Carrying out the above-mentioned research project. Personal data will be kept during the execution of the project and for two more years after its conclusion for scientific validation.
        </p>
        <p style="font-size: 14px; line-height: 1.7;">
          <strong>Legal basis:</strong> Your consent, which can be withdrawn at any time.
        </p>
        <p style="font-size: 14px; line-height: 1.7;">
          <strong>Recipients:</strong> Your personal data will be processed by Universitat Pompeu Fabra, European Commission. Data may be anonymized and published in an open science repository.
        </p>
        <p style="font-size: 14px; line-height: 1.7;">
          <strong>Rights:</strong> You can access your data; request their rectification, deletion, and in certain cases their portability; you may object to their processing and apply for its limitation by following the procedures described at <a href="https://www.upf.edu/web/proteccio-dades/drets" target="_blank">www.upf.edu/web/proteccio-dades/drets</a>. You can contact UPF&rsquo;s Data Protection Officer (<a href="mailto:dpd@upf.edu">dpd@upf.edu</a>) for any queries or if you feel that your rights are not properly respected. Should you not be satisfied, you may file a complaint with the Catalan Data Protection Authority (<a href="https://apdcat.gencat.cat" target="_blank">apdcat.gencat.cat</a>).
        </p>

        <p>Before you start, please consent below.</p>
        </div>
        </div>
    `,
    choices: ['I consent to take part in this study']
};

timeline.push(consentTrial);


/* 1 Demographics: Age, Gender, Microtonal experience */
var demographics = {
  type: jsPsychSurveyHtmlForm,
  data: { trial_name: 'demographics' },
  button_label: 'Continue',
  html: `
    <style>
      .demo-form { text-align: left; max-width: 500px; margin: 0 auto; }
      .demo-form .field-label { font-size: 16px; font-weight: bold; display: block; margin-top: 20px; }
      .demo-form input[type="number"] {
        font-size: 16px; padding: 6px 10px; width: 100px;
        border: 1px solid #ccc; border-radius: 8px;
      }
      .demo-radio-group { margin-top: 8px; }
      .demo-radio-group label {
        font-weight: normal; display: inline-block; margin-right: 20px;
        font-size: 15px; cursor: pointer;
      }
      .demo-radio-group input[type="radio"] { margin-right: 5px; display: inline !important; }
    </style>
    <div class="demo-form">
      <span class="field-label">Age:</span>
      <input type="number" name="age" min="16" max="99" required placeholder="e.g. 28">

      <span class="field-label">Gender:</span>
      <div class="demo-radio-group">
        <label><input type="radio" name="gender" value="Male" required> Male</label>
        <label><input type="radio" name="gender" value="Female"> Female</label>
        <label><input type="radio" name="gender" value="Non-binary"> Non-binary</label>
        <label><input type="radio" name="gender" value="Prefer not to say"> Prefer not to say</label>
      </div>

      <span class="field-label">Have you listened to microtonal music?</span>
      <div class="demo-radio-group">
        <label><input type="radio" name="microtonal" value="yes" required> Yes</label>
        <label><input type="radio" name="microtonal" value="no"> No</label>
      </div>
    </div>
  `
};
timeline.push(demographics);


/* 2 Harmonic question */
// it's evaluated each time this script or function is run.
var randomQuestion = question_simple[Math.floor(Math.random() * question_simple.length)];

var harmonyQuestion = {
    type: jsPsychSurveyMultiChoice,
    stimulus: `harmony`,
    questions: [
        {
            prompt: "The correct functional progression of the previous audio is:",
            options: harmonicOptions,
            required: true,
            name: 'harmonyProgression'
        }
    ],
    preamble: `<div style="margin-bottom: 20px;">Listen to the recording below and then select the correct functional progression. <br>You must listen to the entire recording before you continue:</div>
             <div class="audio-container">
             <audio id="harmonyAudio" controls>
               <source src="${randomQuestion}" type="audio/mpeg">
               Your browser does not support the audio element.
             </audio> </div>`,
    on_load: function() {
        document.querySelector('#jspsych-survey-multi-choice-next').disabled = true;
        document.getElementById('harmonyAudio').addEventListener('ended', function() {
            document.querySelector('#jspsych-survey-multi-choice-next').disabled = false;
        });
    },
    // Include the played audio file in the data to be saved
    data: {
        questionAudio: randomQuestion
    }
};


timeline.push(harmonyQuestion);


/* 3 Music Expertise Level */
var expertiseLevel = {
    type: jsPsychSurveyMultiChoice,
    questions: [
        {
            prompt: ``,
            options: [
                "<strong>Novice:</strong> I recognize music is made of chords but don't understand specific chord roles or names.",
                "<strong>Beginner:</strong> I know basic chord names and can identify major and minor chords but I am less sure about how they function in progressions.",
                "<strong>Intermediate:</strong> I understand chord roles in progressions and can identify subdominant, dominant, and tonic chords, with some knowledge of more complex chords (seventh chords, diminished, augmented).",
                "<strong>Advanced:</strong> I'm proficient in various chord types and their roles in progressions, including modal interchange and secondary dominants.",
                "<strong>Expert:</strong> I have a deep understanding of harmony and can innovate with chord progressions and harmonizations, teaching advanced concepts."
            ],
            required: true,
            horizontal: false,
        }
    ],
    preamble: `<p style="font-size: 26px;"> How would you describe your understanding of harmony?</p>
                <style>
                .jspsych-survey-multi-choice-option { margin-bottom: 20px; margin-right: 200px }
                .jspsych-display-element { padding: 50px !important; } /* Adjust padding as needed */
                </style>`,
    on_finish: function(data) {
        console.log('Expertise Level: ', data.responses);
    }
};

timeline.push(expertiseLevel);


/*----------------------------------------------*/
/*        PART 1: Chord Dissonance Ratings      */
/*----------------------------------------------*/

/* Part 1 Instructions */
var part1Instructions = {
    type: jsPsychHtmlButtonResponse,
    stimulus: `<div style="text-align: left; max-width: 640px; margin: 0 auto;">
               <p style="font-size: 26px; text-align: center; margin-bottom: 20px;">Part 1: Chord Dissonance</p>
               <p style="font-size: 16px; line-height: 1.7;">
                 You will hear <strong>20 isolated chords</strong>.
                 For each chord, rate how <strong>dissonant</strong> it sounds on a scale from 1 to 10.
               </p>
               <p style="font-size: 16px; line-height: 1.7;">
                 <strong>1</strong> = Very consonant (smooth, stable, pleasant)<br>
                 <strong>10</strong> = Very dissonant (rough, tense, clashing)
               </p>
               <p style="font-size: 16px; line-height: 1.7;">
                 Between each chord, a short burst of white noise will play to clear your auditory palette.
                 You must listen to the full chord before submitting your rating.
               </p>
               <p style="font-size: 16px; line-height: 1.7;">
                 Please use headphones if possible.
               </p>
               </div>`,
    choices: ['Continue']
};
timeline.push(part1Instructions);


/* -------------------------------------------------------
   CALIBRATION: Reference chords for dissonance anchoring
   ------------------------------------------------------- */
var calibrationTrial = {
  type: jsPsychHtmlButtonResponse,
  stimulus: `
    <div class="calibration-container">
      <p style="font-size: 22px; text-align: center; margin-bottom: 6px;">Reference Sounds</p>
      <p style="font-size: 15px; text-align: center; color: #555; line-height: 1.6;">
        Before you begin rating, listen to these two reference chords.<br>
        They represent the <strong>extremes</strong> of the dissonance scale you will use.<br>
        Please listen to <strong>both</strong> chords at least once before continuing.
      </p>

      <div style="text-align: center; margin-top: 20px;">

        <div class="calibration-card consonant">
          <div class="card-label" style="color: #2E7D32;">Consonant</div>
          <div class="card-score">This is a <strong>1</strong> on the scale</div>
          <audio id="ref-consonant" controls>
            <source src="${refConsonantFile}" type="audio/wav">
          </audio>
          <div class="calibration-check" id="check-consonant">&#10003; Listened</div>
        </div>

        <div class="calibration-card dissonant">
          <div class="card-label" style="color: #C62828;">Dissonant</div>
          <div class="card-score">This is a <strong>10</strong> on the scale</div>
          <audio id="ref-dissonant" controls>
            <source src="${refDissonantFile}" type="audio/wav">
          </audio>
          <div class="calibration-check" id="check-dissonant">&#10003; Listened</div>
        </div>

      </div>
    </div>
  `,
  choices: ['Begin Part 1'],
  data: {
    trial_name: 'calibration',
    ref_consonant_file: refConsonantFile,
    ref_dissonant_file: refDissonantFile
  },
  on_load: function() {
    /* Disable button until both references have been played */
    var btn = document.querySelector('.jspsych-btn');
    if (btn) { btn.disabled = true; }

    var playedConsonant = false;
    var playedDissonant = false;

    function checkBothPlayed() {
      if (playedConsonant && playedDissonant && btn) {
        btn.disabled = false;
      }
    }

    var audioC = document.getElementById('ref-consonant');
    var audioD = document.getElementById('ref-dissonant');

    if (audioC) {
      audioC.addEventListener('ended', function() {
        playedConsonant = true;
        var mark = document.getElementById('check-consonant');
        if (mark) { mark.classList.add('played'); }
        checkBothPlayed();
      });
    }
    if (audioD) {
      audioD.addEventListener('ended', function() {
        playedDissonant = true;
        var mark = document.getElementById('check-dissonant');
        if (mark) { mark.classList.add('played'); }
        checkBothPlayed();
      });
    }
  }
};
timeline.push(calibrationTrial);


/* Part 1 Trials: 20 chords with white noise between them */
var TOTAL_CHORDS = shuffledChords.length;

for (var i = 0; i < TOTAL_CHORDS; i++) {

  /* White noise palate cleanser: play via HTML audio, auto-advance after it ends */
  (function(idx) {
    timeline.push({
      type: jsPsychHtmlKeyboardResponse,
      stimulus: '<div class="noise-screen">&#9835;</div>' +
                '<audio id="noise-audio" autoplay><source src="00_white_noise_clean.wav" type="audio/wav"></audio>',
      choices: "NO_KEYS",
      trial_duration: 1500,
      data: { trial_name: 'white_noise' }
    });
  })(i);

  /* Chord dissonance rating using SurveyMultiChoice with embedded audio */
  (function(idx) {
    var chord = shuffledChords[idx];
    var trialNum = idx + 1;

    timeline.push({
      type: jsPsychSurveyMultiChoice,
      preamble: '<p class="trial-counter">Chord ' + trialNum + ' of ' + TOTAL_CHORDS + '</p>' +
                '<div class="audio-container">' +
                '<audio id="chord-audio-' + idx + '" controls autoplay>' +
                '<source src="' + chord.file + '" type="audio/wav">' +
                '</audio></div>' +
                '<p style="font-size: 18px; margin-top: 10px;">How <strong>dissonant</strong> is this chord?</p>' +
                '<div style="max-width: 500px; margin: 10px auto 14px;">' +
                  '<div style="display: flex; justify-content: space-between; font-size: 12px; color: #666; margin-bottom: 4px;">' +
                    '<span>Consonant</span><span>Dissonant</span>' +
                  '</div>' +
                  '<div style="height: 10px; border-radius: 5px; background: linear-gradient(to right, #4CAF50, #8BC34A, #CDDC39, #FFEB3B, #FFC107, #FF9800, #FF5722, #F44336);"></div>' +
                  '<div style="display: flex; justify-content: space-between; font-size: 11px; color: #aaa; margin-top: 3px;">' +
                    '<span>1</span><span>5</span><span>10</span>' +
                  '</div>' +
                '</div>',
      questions: [{
        prompt: '',
        options: opt_scale,
        required: true,
        name: 'dissonance',
        horizontal: true
      }],
      on_load: function() {
        var btn = document.querySelector('#jspsych-survey-multi-choice-next');
        if (btn) { btn.disabled = true; }
        var audio = document.getElementById('chord-audio-' + idx);
        if (audio) {
          audio.play().catch(function(e) { console.log('Autoplay blocked:', e); });
          audio.addEventListener('ended', function() {
            var b = document.querySelector('#jspsych-survey-multi-choice-next');
            if (b) { b.disabled = false; }
          });
        }
      },
      on_finish: function(data) {
        data.trial_name     = 'chord_dissonance';
        data.trial_number   = trialNum;
        data.chord_num      = chord.num;
        data.chord_category = chord.cat;
        data.chord_name     = chord.name;
        data.chord_file     = chord.file;
      }
    });
  })(i);
}


/*----------------------------------------------*/
/*        PART 2: Chord Progressions            */
/*----------------------------------------------*/

/* Part 2 Instructions */
var instructions = {
    type: jsPsychHtmlButtonResponse,
    stimulus: `<div style="text-align: left; max-width: 640px; margin: 0 auto;">
               <p style="font-size: 26px; text-align: center; margin-bottom: 20px;">Part 2: Chord Progressions</p>
               <p style="font-size: 16px; line-height: 1.7;">
                 You will hear <strong>10 microtonal chord progressions</strong>.
                 These are not songs or melodies, but harmonic trajectories through microtonal space.
               </p>
               <p style="font-size: 16px; line-height: 1.7;">
                 For each progression, you will rate four perceptual qualities on a scale from 1 to 10.
                 There are no right or wrong answers; simply estimate each value based on your listening impression.
               </p>
               <p style="font-size: 16px; line-height: 1.7;">
                All scales range from <strong>1</strong> (low) to <strong>10</strong> (high).
                Each question includes labeled endpoints to guide your ratings.
              </p>
               </div>`,
    choices: ['Begin Part 2']
};
timeline.push(instructions);


/* Part 2 Audio Questionnaire */
function createSurveyQuestion(i) {
  var audioSource = shuffledPairs[i].audio; // Get the shuffled audio source
  var contextText = shuffledPairs[i].context; // Get the corresponding context
  
  return {
    type: jsPsychSurveyMultiChoice,
    stimulus: `question`+i,
    preamble: `<p class="trial-counter">Progression ${i + 1} of ${shuffledPairs.length}</p>
               <p style="font-size: 14px; font-family: 'Fira Code', monospace; background: #f5f5f5; padding: 10px 18px; border-radius: 8px; display: inline-block; margin-bottom: 10px; word-spacing: 12px;">${contextText}</p>
               <p class="small-text">Play the entire recording before answering</p>
               <div class="audio-container">
                 <audio id="audio-${i}" controls autoplay>
                   <source src="${audioSource}" type="audio/mpeg">
                 </audio>
               </div>`,    
    questions: [
              {
                prompt: '<p class="questionStyles;">1) <strong>Harmony</strong>: How coherent do you find the harmonic motion?</p>' +
                        '<div style="max-width: 560px; margin: 6px auto 8px;">' +
                          '<div style="display: flex; justify-content: space-between; font-size: 11px; color: #666;">' +
                            '<span>1 = Not coherent at all</span><span>10 = Very coherent</span>' +
                          '</div>' +
                          '<div style="height: 8px; border-radius: 4px; background: linear-gradient(to right, #F44336, #FF9800, #FFEB3B, #8BC34A, #4CAF50);"></div>' +
                        '</div>',
                options: opt_scale,
                required: true,
                name: `harmony`,
                horizontal: true,
              },
              {
                prompt: '<p class="questionStyles;">2) <strong>Musicality</strong>: How musical is the chord progression?</p>' +
                        '<div style="max-width: 560px; margin: 6px auto 8px;">' +
                          '<div style="display: flex; justify-content: space-between; font-size: 11px; color: #666;">' +
                            '<span>1 = Not musical at all</span><span>10 = Very musical</span>' +
                          '</div>' +
                          '<div style="height: 8px; border-radius: 4px; background: linear-gradient(to right, #F44336, #FF9800, #FFEB3B, #8BC34A, #4CAF50);"></div>' +
                        '</div>',
                options: opt_scale,
                required: true,
                name: `musicality`,
                horizontal: true,
              },
              {
                prompt: '<p class="questionStyles;">3) <strong>Dissonance</strong>: How dissonant is this chord progression?</p>' +
                        '<div style="max-width: 560px; margin: 6px auto 8px;">' +
                          '<div style="display: flex; justify-content: space-between; font-size: 11px; color: #666;">' +
                            '<span>1 = Very consonant</span><span>10 = Very dissonant</span>' +
                          '</div>' +
                          '<div style="height: 8px; border-radius: 4px; background: linear-gradient(to right, #4CAF50, #8BC34A, #CDDC39, #FFEB3B, #FFC107, #FF9800, #FF5722, #F44336);"></div>' +
                        '</div>',
                options: opt_scale,
                required: true,
                name: `dissonance`,
                horizontal: true,
              },
              {
                prompt: '<p class="questionStyles;">4) <strong>Novelty</strong>: How surprising or novel do you find this progression?</p>' +
                        '<div style="max-width: 560px; margin: 6px auto 8px;">' +
                          '<div style="display: flex; justify-content: space-between; font-size: 11px; color: #666;">' +
                            '<span>1 = Very familiar</span><span>10 = Very novel</span>' +
                          '</div>' +
                          '<div style="height: 8px; border-radius: 4px; background: linear-gradient(to right, #90CAF9, #42A5F5, #1E88E5, #1565C0, #0D47A1);"></div>' +
                        '</div>',
                options: opt_scale,
                required: true,
                name: `novelty`,
                horizontal: true,
              }
            ],
    on_finish: function(data) {
      data.questionAudio = audioSource;
      data.progression = contextText;
      data.trial_name = 'chord_progression';
    },
    on_load: function() {
      let nextButton = document.querySelector('input[type="submit"]');
      if(nextButton) {
        nextButton.disabled = true;
      }

      let audioEnded = false;

      function checkReady() {
        if (!audioEnded) return;
        let answered = document.querySelectorAll('.jspsych-survey-multi-choice-question input[type="radio"]:checked').length;
        if (answered >= 4 && nextButton) {
          nextButton.disabled = false;
        }
      }

      let audioElement = document.getElementById(`audio-${i}`);
      if(audioElement) {
        audioElement.play().catch(function(e) { console.log('Autoplay blocked:', e); });
        audioElement.addEventListener('ended', function() {
          audioEnded = true;
          checkReady();
        });
      }

      document.querySelectorAll('.jspsych-survey-multi-choice-question input[type="radio"]').forEach(function(radio) {
        radio.addEventListener('change', checkReady);
      });
    }
  };
}

for (let i = 0; i < shuffledPairs.length; i++) {
  /* White noise palate cleanser before every progression */
  timeline.push({
    type: jsPsychHtmlKeyboardResponse,
    stimulus: '<div class="noise-screen">&#9835;</div>' +
              '<audio id="noise-audio-p2" autoplay><source src="00_white_noise_clean.wav" type="audio/wav"></audio>',
    choices: "NO_KEYS",
    trial_duration: 1500,
    data: { trial_name: 'white_noise' }
  });

  timeline.push(createSurveyQuestion(i));
}

/* ------------------------------ End ------------------------------ */
    var exit_survey = {
    type: jsPsychSurveyHtmlForm,
    data: {trial_name: 'exit_survey'},
    button_label: 'Finish and Send',
    html: 
        '<style>p {text-align:left; spellcheck=false;} input[type="text"] {width:8ch;} fieldset {border:1px solid #999;box-shadow:2px 2px 5px #999;} legend {background:#fff;text-align:left;font-size:110%;}</style>'+
        '<p style="font-size: 28px; text-align: center;">Thank you!</p>'+
        '<p style="font-size: 16px; line-height: 1.7; text-align: center;">Thanks a lot for your participation, we appreciate it sincerely.</p>'+
        '<p>Would you like to share any thoughts or comments about the microtonal harmonies you listened to?</p>'+
        '<p style="text-align:left;"><textarea name="exit_comments" rows="3" style="width:90%;"></textarea></p>'+
        '<p>If you experienced any difficulties during the study, please describe them here.</p>'+
        '<p style="text-align:left;"><textarea name="exit_difficulties" rows="3" style="width:90%;"></textarea></p>'+
        '<div style="background-color: #FF5722; color: white; padding: 15px 25px; border-radius: 12px; font-size: 18px; font-weight: bold; text-align: center; margin-top: 25px;">'+
        '&#9888; Click "Finish and Send" below to submit your data. Without this step your participation will NOT be recorded.'+
        '</div>'+
        '',
    };
    timeline.push(exit_survey);
    
jsPsych.run(timeline);