/* ============================================================
   ANIMA – Microtonal Listening Test (v2, 24 clips)
   ------------------------------------------------------------
   24 audio clips = 8 dataset + 8 model_A + 8 model_B,
   spanning 4 styles (jazz, blues, bossa, rock) x 8 chord
   transformations.  Each clip is rated on four Likert-1..10
   scales: Harmony, Musicality, Dissonance, Novelty.
   ------------------------------------------------------------
   Drop this file into cognition.run together with the 24 .mp3
   clips (flat, no sub-folders), mtg_logo.png, and the white
   noise file 00_white_noise_clean.wav.
   ============================================================ */

/* const success_url = 'https://app.prolific.co/submissions/complete?cc=XXXXXXX'; */

var timeline = [];

function addStyles(styles) {
  var styleSheet = document.createElement("style");
  styleSheet.type = "text/css";
  styleSheet.innerText = styles;
  document.head.appendChild(styleSheet);
}

var opt_scale = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'];

/* ------------------------------------------------------------
   THE 24 CLIPS
   Filenames already include source + style + transformation so
   nothing else needs to be kept in sync.  The order below does
   not matter – we shuffle before use.
   ------------------------------------------------------------ */
var all_audio_samples = [
  "01_dataset_01_Autumn_Leaves__jazz__type_4_upmajor.mp3",
  "02_dataset_02_Misty__jazz__type_5_minor.mp3",
  "03_dataset_03_Crossroads__blues__type_2_subminor.mp3",
  "04_dataset_04_Bessies_Blues__blues__type_6_neutral_n.mp3",
  "05_dataset_05_Wave__bossa__type_1_neutral.mp3",
  "06_dataset_06_So_Tinha_De_Ser_Com_Voce__bossa__type_3_major.mp3",
  "07_dataset_07_I_Cant_Help_It__rock__type_5_major_v2.mp3",
  "08_dataset_08_Rock_With_You__rock__type_4_minor.mp3",
  "09_model_A_01_jazz__type_5_major_v2.mp3",
  "10_model_A_02_jazz__type_5_minor.mp3",
  "11_model_A_03_blues__type_2_subminor.mp3",
  "12_model_A_04_blues__type_6_neutral_n.mp3",
  "13_model_A_05_bossa__type_1_neutral.mp3",
  "14_model_A_06_bossa__type_3_major.mp3",
  "15_model_A_07_rock__type_4_upmajor.mp3",
  "16_model_A_08_rock__type_4_minor.mp3",
  "17_model_B_01_jazz__type_5_major_v2.mp3",
  "18_model_B_02_jazz__type_5_minor.mp3",
  "19_model_B_03_blues__type_2_subminor.mp3",
  "20_model_B_04_blues__type_6_neutral_n.mp3",
  "21_model_B_05_bossa__type_1_neutral.mp3",
  "22_model_B_06_bossa__type_3_major.mp3",
  "23_model_B_07_rock__type_4_upmajor.mp3",
  "24_model_B_08_rock__type_4_minor.mp3"
];

/* Parse "NN_<source>_<idx>_<title?>__<style>__type_<type>.mp3"
   Source is "dataset", "model_A" or "model_B". */
function parseClipMeta(fname) {
  var stem = fname.replace(/\.(wav|mp3)$/i, '');
  // Identify source (longest match wins so model_A isn't shadowed)
  var source = null;
  if (stem.indexOf('model_A') !== -1) source = 'model_A';
  else if (stem.indexOf('model_B') !== -1) source = 'model_B';
  else if (stem.indexOf('dataset') !== -1) source = 'dataset';
  // style is whatever appears immediately before "__type_"
  var m = stem.match(/__([a-z0-9]+)__type_([A-Za-z0-9_]+)$/i);
  return {
    source: source,
    style: m ? m[1] : null,
    type: m ? ('type_' + m[2]) : null
  };
}

/* Build the shuffled sample list with metadata attached */
var clipEntries = all_audio_samples.map(function (f) {
  var meta = parseClipMeta(f);
  return { audio: f, source: meta.source, style: meta.style, type: meta.type };
});

/* ============================================================
   PRELOAD
   ============================================================ */
var allPreloadAudio = [].concat(all_audio_samples, ["00_white_noise_clean.wav"]);
console.log("=== FILES TO PRELOAD (" + allPreloadAudio.length + ") ===");
console.log(allPreloadAudio);

var preload = {
  type: jsPsychPreload,
  audio: allPreloadAudio,
  continue_after_error: true,
  on_finish: function (data) {
    console.log("=== PRELOAD RESULTS ===");
    console.log("Success:", data.success);
    console.log("Failed audio:", data.failed_audio);
  }
};
timeline.push(preload);

/* ============================================================
   STYLES  (kept identical to v1 minus the chord-calibration CSS)
   ============================================================ */
const fontStyles = `
body, .jspsych-display-element, .jspsych-content {
  font-family: 'Fira Code', 'Monaco', 'SF Mono', 'Consolas', monospace;
}
`;
var fontLink = document.createElement("link");
fontLink.href = "https://fonts.googleapis.com/css2?family=Fira+Code:wght@300;400;500;600;700&display=swap";
fontLink.rel = "stylesheet";
document.head.appendChild(fontLink);

const buttonStyles = `
.jspsych-btn {
  background-color: #ffaa00;
  border: none; color: white;
  padding: 10px 24px;
  font-size: 16px; margin: 4px 2px;
  cursor: pointer; border-radius: 12px;
  transition: background 0.3s, transform 0.3s;
}
.jspsych-btn:hover { background-color: #ffaa00; transform: scale(1.025); }
`;

const preambleStyles = `
.preamble-text { font-size: 24px; }
.small-text    { font-size: 16px; }
.audio-container {
  background-color: #0099FF99;
  display: inline-block;
  margin: 20px 20px 50px;
  padding: 5px;
  border-radius: 15px;
  border-bottom: 2px solid #0003;
}
`;

const questionStyles = `
.jspsych-survey-multi-choice-question:after, .jspsych-survey-text-question:after,
.jspsych-survey-multi-choice-question::after, .jspsych-survey-text-question::after,
.jspsych-survey-multi-choice-question .required,
p.jspsych-survey-multi-choice-question span.required {
  content: none !important; display: none !important;
}
`;

const scaleButtonStyles = `
.jspsych-survey-multi-choice .jspsych-btn,
.jspsych-survey-multi-choice-horizontal .jspsych-btn {
  font-size: 25px; padding: 5px 10px;
  background-color: #007bff; color: #ffffff;
}
.jspsych-survey-multi-choice .jspsych-btn:hover,
.jspsych-survey-multi-choice-horizontal .jspsych-btn:hover {
  background-color: #0056b3;
}
`;

const radialButtonStyles = `
.jspsych-survey-multi-choice-option { display: inline-block; margin-right: 15px; }
.jspsych-survey-multi-choice-option input[type="radio"] { display: none; }
.jspsych-survey-multi-choice-option label {
  background-color: #f0f0f0;
  padding: 5px 20px 5px;
  border-radius: 20px;
  font-size: 18px; cursor: pointer;
  display: inline-block;
  transition: background-color 0.3s, color 0.3s;
}
.jspsych-survey-multi-choice-option label:has(> input[type="radio"]:checked) {
  background-color: #ffaa00; color: white;
}
`;

const partStyles = `
.trial-counter {
  font-size: 14px; color: #666; margin-bottom: 6px;
}
.noise-screen {
  font-size: 24px; color: #999; padding: 40px;
}

/* ---- slider question block ---- */
.slider-block {
  max-width: 620px; margin: 18px auto 24px; text-align: left;
}
.slider-block .q-prompt {
  font-size: 16px; margin-bottom: 4px;
}
.slider-block .q-endpoints {
  display: flex; justify-content: space-between;
  font-size: 11px; color: #666; margin: 2px 4px;
}
.slider-block input[type="range"] {
  -webkit-appearance: none; appearance: none;
  width: 100%; height: 8px; border-radius: 4px;
  background: linear-gradient(to right, #F44336, #FF9800, #FFEB3B, #8BC34A, #4CAF50);
  outline: none; margin: 4px 0 0;
}
.slider-block.dissonance input[type="range"] {
  background: linear-gradient(to right, #4CAF50, #8BC34A, #CDDC39, #FFEB3B, #FFC107, #FF9800, #FF5722, #F44336);
}
.slider-block.novelty input[type="range"] {
  background: linear-gradient(to right, #90CAF9, #42A5F5, #1E88E5, #1565C0, #0D47A1);
}
.slider-block input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none; appearance: none;
  width: 22px; height: 22px; border-radius: 50%;
  background: #ffaa00; border: 2px solid #fff;
  box-shadow: 0 0 0 1px #888; cursor: pointer;
}
.slider-block input[type="range"]::-moz-range-thumb {
  width: 22px; height: 22px; border-radius: 50%;
  background: #ffaa00; border: 2px solid #fff;
  box-shadow: 0 0 0 1px #888; cursor: pointer;
}
.slider-block.untouched input[type="range"]::-webkit-slider-thumb { opacity: 0.35; }
.slider-block.untouched input[type="range"]::-moz-range-thumb     { opacity: 0.35; }
.slider-block .q-readout {
  display: inline-block; margin-top: 6px;
  font-size: 14px; color: #444;
  background: #f5f5f5; padding: 2px 10px; border-radius: 10px;
}
.slider-block.untouched .q-readout {
  color: #aaa; font-style: italic;
}
.slider-scale-ticks {
  display: flex; justify-content: space-between;
  font-size: 10px; color: #aaa; margin: 2px 4px 0;
}
`;

addStyles(`${fontStyles} ${buttonStyles} ${scaleButtonStyles} ${preambleStyles} ${questionStyles} ${radialButtonStyles} ${partStyles}`);

/* ============================================================
   INIT
   ============================================================ */
var jsPsych = initJsPsych({
  show_progress_bar: true,
  override_safe_mode: true,
  on_finish: function () {
    /* window.location.href = success_url; */
    jsPsych.getDisplayElement().innerHTML = `
      <div style="
        font-family: 'Fira Code', 'Monaco', 'SF Mono', 'Consolas', monospace;
        max-width: 640px; margin: 80px auto; text-align: center; padding: 0 24px;
      ">
        <div style="
          background: linear-gradient(135deg, #0afa99 0%, #0099FF44 100%);
          border-radius: 24px; padding: 48px 40px;
          box-shadow: 0 4px 24px #0002;
        ">
          <p style="font-size: 48px; margin: 0 0 8px;">&#127925;</p>
          <p style="font-size: 30px; font-weight: 600; margin: 0 0 16px; color: #111;">
            Thank you very much!
          </p>
          <p style="font-size: 17px; line-height: 1.8; color: #333; margin: 0 0 28px;">
            Your responses have been recorded successfully.<br>
            We sincerely appreciate your time and careful listening.
          </p>
          <div style="
            background: #fff; border-radius: 14px;
            padding: 16px 24px; display: inline-block;
            border: 2px solid #ffaa00; margin-bottom: 28px;
          ">
            <p style="font-size: 14px; color: #555; margin: 0; line-height: 1.7;">
              &#10003;&nbsp; All 24 clip ratings submitted<br>
              &#10003;&nbsp; Data saved to Cognition.run<br>
              &#10003;&nbsp; You may now close this window
            </p>
          </div>
          <p style="font-size: 13px; color: #777; margin: 0;">
            ANIMA &mdash; Artificial INtelligence-based Interactive Microtonal Compositional Assistant<br>
            Music Technology Group, Universitat Pompeu Fabra
          </p>
        </div>
      </div>`;
  }
});

/* Fully random playback of all 24 clips.
   Explicit Fisher-Yates shuffle over the combined pool (dataset + model_A +
   model_B) so presentation order is decoupled from source grouping. */
function fisherYatesShuffle(arr) {
  var a = arr.slice();
  for (var i = a.length - 1; i > 0; i--) {
    var j = Math.floor(Math.random() * (i + 1));
    var tmp = a[i]; a[i] = a[j]; a[j] = tmp;
  }
  return a;
}
var shuffledClips = fisherYatesShuffle(clipEntries);
console.log("Shuffled clip order (all 24, random, sources interleaved):");
console.log(shuffledClips.map(function (c, i) {
  return (i + 1) + '. ' + c.source + '  ' + c.style + '  ' + c.type;
}).join('\n'));

/* ============================================================
   1. CONSENT
   ============================================================ */
var consentTrial = {
  type: jsPsychHtmlButtonResponse,
  stimulus: `
        <div style="max-width: 740px; margin: 0 auto;">
          <div style="text-align: left; margin-bottom: 20px;">
            <img src="mtg_logo.png" alt="MTG - Music Technology Group" style="max-width: 150px; display: block; margin-bottom: 0;">
          </div>
          <p style="font-size: 25px; background-color: #0afa; border-radius: 20px; padding: 20px; margin-top: 0;">Welcome to the microtonal chord-progression study</p>

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

            <p style="font-size: 16px;">You have been invited to take part in a listening study focused on <strong>microtonal chord progressions</strong>.
              <br><strong>The following questionnaire is only compatible with Google Chrome and Safari.</strong>
              <br>The study lasts about 20 minutes. This study is part of a research project funded by MSCA European Union, Grant Agreement ID: 101203318.
            </p>

            <p style="font-size: 20px;">Read the following information carefully before proceeding.</p>
            <ul style="font-size: 16px;">
              <li>Your participation is unpaid.</li>
              <li>Your participation is confidential and anonymous. The data you provide are strictly for academic research.</li>
              <li>There are no risks identified in relation to this study.</li>
              <li>This study has been reviewed by a project-specific ethics committee organised by the University Ethical Committee, which has concluded that the personal data collected does not raise ethical concerns.</li>
              <li>If you have questions or concerns after the study, please feel free to contact <a href="mailto:david.cabrera@upf.edu">david.cabrera@upf.edu</a>.</li>
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
        </div>`,
  choices: ['I consent to take part in this study']
};
timeline.push(consentTrial);

/* ============================================================
   2. DEMOGRAPHICS
   ============================================================ */
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

      <span class="field-label">Have you listened to microtonal music before?</span>
      <div class="demo-radio-group">
        <label><input type="radio" name="microtonal" value="yes" required> Yes</label>
        <label><input type="radio" name="microtonal" value="no"> No</label>
      </div>
    </div>
    `
};
timeline.push(demographics);

/* ============================================================
   3. EXPERTISE LEVEL
   ============================================================ */
var expertiseLevel = {
  type: jsPsychSurveyMultiChoice,
  data: { trial_name: 'expertise' },
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
      name: 'expertise'
    }
  ],
  preamble: `<p style="font-size: 26px;">How would you describe your understanding of harmony?</p>
               <style>
                 .jspsych-survey-multi-choice-option { margin-bottom: 20px; margin-right: 200px }
                 .jspsych-display-element { padding: 50px !important; }
               </style>`
};
timeline.push(expertiseLevel);

/* ============================================================
   4. MAIN TASK — 24 CLIPS, 4 QUESTIONS EACH
   ============================================================ */
var instructions = {
  type: jsPsychHtmlButtonResponse,
  stimulus: `<div style="text-align: left; max-width: 640px; margin: 0 auto;">
                 <p style="font-size: 26px; text-align: center; margin-bottom: 20px;">Listening task</p>
                 <p style="font-size: 16px; line-height: 1.7;">
                   You will hear <strong>24 short microtonal chord progressions</strong>,
                   each <strong>16 bars long</strong>. These are not songs or melodies,
                   but harmonic trajectories through microtonal space.
                 </p>
                 <p style="font-size: 16px; line-height: 1.7;">
                   For each progression you will rate the <strong>same four perceptual qualities</strong>
                   on a scale from 1 to 10. There are no right or wrong answers &mdash;
                   simply estimate each value based on your listening impression.
                 </p>
                 <p style="font-size: 16px; line-height: 1.7; background: #fff8e1; border-left: 4px solid #ffaa00; padding: 12px 16px; border-radius: 6px;">
                   <strong>Important:</strong> some individual chords may sound unusual or unfamiliar
                   to your ear &mdash; this is expected, as the audio samples use a microtonal tuning system.
                   Please <strong>rate the progression as a whole</strong>, not a single chord in isolation.
                   Try to listen to the overall harmonic motion and the "big picture" rather than
                   penalising a clip because one chord sounded strange.
                 </p>
                 <p style="font-size: 16px; line-height: 1.7;">
                   A short burst of white noise plays between clips to clear your auditory palette.
                   You must listen to each clip fully before submitting your ratings.
                 </p>
                 <p style="font-size: 16px; line-height: 1.7;">
                   Please use headphones if possible.
                 </p>
               </div>`,
  choices: ['Begin']
};
timeline.push(instructions);

function createTrial(i) {
  var clip = shuffledClips[i];
  var trialNum = i + 1;
  var total = shuffledClips.length;

  /* Build the 4 slider blocks inline.  Sliders cover [1.00, 10.00]
     with step 0.01 so we capture continuous perceptual ratings.
     Each slider starts at 5.5 (centre) but is flagged "untouched"
     until the participant actually interacts with it, which keeps
     the Submit button disabled. */
  var questions = [
    {
      key: 'harmony', label: 'Harmony', title: 'How coherent do you find the harmonic (Voicing) motion?',
      lo: 'Not coherent at all', hi: 'Very coherent', cls: 'harmony'
    },
    {
      key: 'plausibility', label: 'Musicality', title: 'How plausible is this chord progression for a potential song?',
      lo: 'Not plausible', hi: 'Very plausible', cls: 'plausibility'
    },
    {
      key: 'dissonance', label: 'Dissonance', title: 'How dissonant is this chord progression?',
      lo: 'Very consonant', hi: 'Very dissonant', cls: 'dissonance'
    },
    {
      key: 'novelty', label: 'Novelty', title: 'How surprising or novel do you find this progression?',
      lo: 'Very familiar', hi: 'Very novel', cls: 'novelty'
    }
  ];

  var sliderHtml = questions.map(function (q, qi) {
    return (
      '<div class="slider-block ' + q.cls + ' untouched" data-key="' + q.key + '">' +
      '<p class="q-prompt">' + (qi + 1) + ') <strong>' + q.label + '</strong>: ' + q.title + '</p>' +
      '<div class="q-endpoints"><span>1 = ' + q.lo + '</span><span>10 = ' + q.hi + '</span></div>' +
      '<input type="range" name="' + q.key + '" min="1" max="10" step="0.01" value="5.5" ' +
      'data-touched="false" aria-label="' + q.label + '">' +
      '<div class="slider-scale-ticks"><span>1</span><span>5</span><span>10</span></div>' +
      '<div class="q-readout" id="readout-' + i + '-' + q.key + '">Drag to rate</div>' +
      '</div>'
    );
  }).join('');

  return {
    type: jsPsychSurveyHtmlForm,
    button_label: 'Submit ratings',
    preamble: `<p class="trial-counter">Clip ${trialNum} of ${total}</p>
                   <p class="small-text">Play the entire recording before answering.</p>
                   <div class="audio-container">
                     <audio id="audio-${i}" controls autoplay>
                       <source src="${clip.audio}" type="audio/mpeg">
                       Your browser does not support the audio element.
                     </audio>
                   </div>`,
    html: sliderHtml +
      '<div id="trial-status-' + i + '" class="trial-status" ' +
      'style="margin:10px auto 0;font-size:13px;color:#b36b00;' +
      'background:#fff8e1;padding:8px 14px;border-radius:8px;' +
      'display:inline-block;border:1px solid #ffcc66;">' +
      '&#9432; Please listen to the full clip and move all four sliders before submitting.' +
      '</div>',
    data: {
      trial_name: 'clip_rating',
      trial_number: trialNum,
      clip_audio: clip.audio,
      clip_source: clip.source,
      clip_style: clip.style,
      clip_type: clip.type
    },
    on_load: function () {
      var audioEnded = false;
      var form = document.querySelector('#jspsych-survey-html-form')
        || document.querySelector('form');
      var submitBtn = document.querySelector('#jspsych-survey-html-form-next')
        || (form && form.querySelector('input[type="submit"],button[type="submit"]'));
      var statusEl = document.getElementById('trial-status-' + i);

      function allTouched() {
        var sliders = document.querySelectorAll('.slider-block input[type="range"]');
        for (var s = 0; s < sliders.length; s++) {
          if (sliders[s].dataset.touched !== 'true') return false;
        }
        return sliders.length > 0;
      }
      function ready() { return audioEnded && allTouched(); }
      function refresh() {
        var ok = ready();
        if (submitBtn) submitBtn.disabled = !ok;
        if (statusEl) {
          if (ok) {
            statusEl.style.display = 'none';
          } else {
            statusEl.style.display = 'inline-block';
            var msgs = [];
            if (!audioEnded) msgs.push('listen to the full clip');
            if (!allTouched()) msgs.push('move all four sliders');
            statusEl.innerHTML = '&#9432; Please ' + msgs.join(' and ') + ' before submitting.';
          }
        }
      }

      /* Block form submission while not ready (covers Enter-key submits
         and any case where disabled attribute is bypassed). */
      if (form) {
        form.addEventListener('submit', function (ev) {
          if (!ready()) { ev.preventDefault(); ev.stopPropagation(); refresh(); }
        }, true);
      }

      /* Wire up sliders: live readout + mark as touched on first interaction */
      document.querySelectorAll('.slider-block').forEach(function (block) {
        var slider = block.querySelector('input[type="range"]');
        var readout = block.querySelector('.q-readout');
        function touch() {
          if (slider.dataset.touched !== 'true') {
            slider.dataset.touched = 'true';
            block.classList.remove('untouched');
          }
          readout.textContent = parseFloat(slider.value).toFixed(2);
          refresh();
        }
        slider.addEventListener('input', touch);
        slider.addEventListener('change', touch);
      });

      /* Wire up audio gating */
      var audioElement = document.getElementById('audio-' + i);
      if (audioElement) {
        var markEnded = function () { audioEnded = true; refresh(); };
        audioElement.addEventListener('ended', markEnded);
        /* Fallback: some browsers don't fire 'ended' reliably on short mp3s */
        audioElement.addEventListener('timeupdate', function () {
          if (audioElement.duration && audioElement.currentTime >= audioElement.duration - 0.05) {
            markEnded();
          }
        });
        var playPromise = audioElement.play();
        if (playPromise && playPromise.catch) {
          playPromise.catch(function (e) {
            console.log('Autoplay blocked, waiting for user to press play:', e);
          });
        }
      } else {
        /* No audio element found -> don't lock participant out */
        audioEnded = true;
      }

      refresh();
    },
    on_finish: function (data) {
      /* jsPsychSurveyHtmlForm stores submitted values in data.response
         as strings — coerce to floats for downstream analysis. */
      if (data.response) {
        data.harmony_raw = data.response.harmony;
        data.plausibility_raw = data.response.plausibility;
        data.dissonance_raw = data.response.dissonance;
        data.novelty_raw = data.response.novelty;
        data.harmony = parseFloat(data.response.harmony);
        data.plausibility = parseFloat(data.response.plausibility);
        data.dissonance = parseFloat(data.response.dissonance);
        data.novelty = parseFloat(data.response.novelty);
      }
    }
  };
}

for (let i = 0; i < shuffledClips.length; i++) {
  /* White-noise palette cleanser between clips */
  timeline.push({
    type: jsPsychHtmlKeyboardResponse,
    stimulus: '<div class="noise-screen">&#9835;</div>' +
      '<audio id="noise-audio" autoplay>' +
      '<source src="00_white_noise_clean.wav" type="audio/wav"></audio>',
    choices: "NO_KEYS",
    trial_duration: 1500,
    data: { trial_name: 'white_noise' }
  });

  timeline.push(createTrial(i));
}

/* ============================================================
   5. EXIT SURVEY
   ============================================================ */
var exit_survey = {
  type: jsPsychSurveyHtmlForm,
  data: { trial_name: 'exit_survey' },
  button_label: 'Finish and Send',
  html:
    '<style>p{text-align:left;}input[type="text"]{width:8ch;}fieldset{border:1px solid #999;box-shadow:2px 2px 5px #999;}legend{background:#fff;text-align:left;font-size:110%;}</style>' +
    '<p style="font-size: 28px; text-align: center;">Thank you!</p>' +
    '<p style="font-size: 16px; line-height: 1.7; text-align: center;">Thanks a lot for your participation, we appreciate it sincerely.</p>' +
    '<p>Would you like to share any thoughts or comments about the microtonal harmonies you listened to?</p>' +
    '<p style="text-align:left;"><textarea name="exit_comments" rows="3" style="width:90%;"></textarea></p>' +
    '<p>If you experienced any difficulties during the study, please describe them here.</p>' +
    '<p style="text-align:left;"><textarea name="exit_difficulties" rows="3" style="width:90%;"></textarea></p>' +
    '<div style="background-color:#FF5722;color:white;padding:15px 25px;border-radius:12px;font-size:18px;font-weight:bold;text-align:center;margin-top:25px;">' +
    '&#9888; Click "Finish and Send" below to submit your data. Without this step your participation will NOT be recorded.' +
    '</div>'
};
timeline.push(exit_survey);

jsPsych.run(timeline);