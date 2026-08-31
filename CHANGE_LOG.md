# QIQ Paper Checker Change Log

This document records the refinements made to QIQ Paper Checker to make it clearer, calmer, and easier for teachers and faculty members to use.

## Overall direction

- Reworked the interface around a teacher's marking workflow rather than developer or system terminology.
- Increased readability with larger, more consistent typography and improved spacing.
- Reduced the alarm-dashboard appearance by removing repeated warning symbols, technical statuses, and unnecessary diagnostic information.
- Kept teacher decisions authoritative: marks entered by the educator become the final marks used throughout the application and report.

## Upload and setup experience

- Reorganised the setup into clear sections for the question paper, reference material, student answer sheet, student details, and marking guidance.
- Numbered the four setup sections consistently: question paper, reference material, student answer sheet, and student details.
- Added a small amount of breathing room above the first **Question paper** section while keeping it visually connected to the panel.
- Improved upload descriptions to explain accepted documents and handwriting support in plain language.
- Removed duplicate numbers and upload symbols from inside the upload areas so the section headings provide the only numbering.
- Standardised all three upload areas to the same height, spacing, typography, and **Browse files** action.
- Added clear confirmation messages after successful uploads.
- Changed upload actions to **Add more files** after a document has been added.
- Made completed upload areas more compact so later setup sections remain easier to reach.
- Standardised the term **Student answer sheet** throughout the application.
- Made reference material clearly optional and explained when marking uses subject knowledge instead.
- Improved question-mark entry so sub-parts remain grouped and visible while marks are entered.
- Kept questions in place after their marks are entered instead of moving them elsewhere.
- Added a sticky primary action so starting the paper check is easier to access.
- Moved the sticky **Start checking paper** action into an opaque footer with a clear top boundary so it never covers form content.
- Kept the action in normal document flow on tablet and mobile layouts.
- Rewrote the new-paper guidance in natural educator language, including **Prepare the paper for checking** and **Ready when you are** states.
- Clarified the three setup items as recommended question-paper support, optional reference guidance, and the required student work.

## Answer and marking guide

- Renamed **Expected answer / marking scheme** to the clearer **Answer and marking guide**.
- Made the answer and marking guide available even when a question paper has been uploaded.
- Allowed educators to add expected points, acceptable alternatives, and mark-allocation instructions.
- Kept educator guidance optional when the question paper already provides the questions and marks.
- Retained **Use sample** to demonstrate the expected marking-guidance format.
- Renamed the repeated action from **Re-grade only** to the clearer **Mark again**.

## Student details

- Made the student-details card collapsible to reduce setup-page length.
- Assigned it setup number **4** to continue the sequence used by the upload sections.
- Replaced the text-like disclosure symbol with a cleaner circular chevron.
- Kept a clear card boundary around the section so its expandable content remains visually grouped.
- Added labelled fields for:
  - Name
  - Roll No.
  - Subject
- Added flexible student-information rows that educators can create or delete.
- Replaced mechanical labels such as **Field** and **Value** with natural prompts such as **For example, Class or Section** and **Enter details**.
- Included standard and custom student details in the final report.

## Marks setup

- Removed the pre-filled total of **20 marks** when no question paper is supplied.
- The total-marks field now starts empty and asks the teacher to enter the correct maximum mark.
- Kept the answer-and-marking-guide and total-marks labels naturally aligned with their full-width input fields.

## Mark review experience

- Renamed the main student-paper tab to **Student Answer Sheet**.
- Added a larger and more visible question navigator.
- Kept question numbers readable and grouped related sub-parts logically.
- Consolidated repeated coverage and warning panels into one calm **Review summary**.
- Removed repeated success messages and page-placement warnings from every question card.
- Replaced technical and mechanical terms with teacher-facing language.
- Changed **Marking reason** to **Marking Analysis**.
- Moved the student's answer before the marking explanation.
- Opened the student's answer by default so the teacher sees the response before reviewing the awarded mark.
- Added a quotation-style visual marker for the student's answer instead of a mechanical warning or disclosure icon.
- Changed **View answer on page** to **View on answer sheet**.
- Redesigned the final-mark control to be compact and properly spaced.
- Added a brief **Mark updated** confirmation after a teacher changes a mark.
- Removed developer-only raw response and reasoning controls from the teacher interface.

## Clear question and reference states

- Simplified unanswered-question messaging to **Question not attempted**.
- Removed phrases such as **No page location available** and **No answer was confidently detected**.
- Distinguished between an unanswered question and writing that could not be matched confidently.
- Added clear combined states:
  - Answer found with no reference material: the mark was based on subject knowledge.
  - Question not attempted with no reference material: both conditions are stated.
  - Question not attempted with reference material available: only the unanswered state is shown.
- Distinguished between **No reference material was provided** and **Reference did not cover this question**.
- Applied the same state wording in the review screen, answer-sheet view, and final report.

## Teacher-friendly language

- Replaced terms such as OCR, extracted text, evaluation, model confidence, evaluator response, pipeline, and grounding in visible teacher-facing areas.
- Renamed **Check Extracted Text** to **Answer Text**.
- Replaced **AI confidence** with quieter teacher-facing review language.
- Replaced **General knowledge** with **Used subject knowledge**.
- Replaced **Examiner-set** with **Set by teacher** or **Mark changed**.
- Simplified processing and service messages so they describe what the teacher needs to know.
- Updated the workflow stages to:
  - Upload
  - Answer Sheet Read
  - Marks Awarded
  - Results Prepared

## Workflow feedback

- Added a spinner to the stage currently being processed.
- Kept completed stages visually distinct and future stages numbered.
- Ensured **Results Prepared** completes once the marking review is available.
- Improved progress wording during reading, marking, and report preparation.
- Kept the detailed workflow stages visible during setup and processing, when they help explain what is happening.
- Replaced the completed four-stage tracker with a calm **Marks ready for review** status once marking finishes.
- Added the available student name, subject, and current marks to the completed header status for useful classroom context.
- Reduced the visual competition between the workflow status and the Light/Dark control.

## Final report

- Changed the report description to **Teacher-reviewed descriptive assessment**.
- Removed model names and low-level implementation details from the report footer.
- Simplified pending-mark and teacher-adjustment explanations.
- Added standard and custom student-information fields to the report.
- Added a final reminder to check marks and student details before printing or sharing.
- Simplified the print action to **Print report**.
- Preserved a clean black-on-white print layout.

## Themes and visual design

- Added Light and Dark themes.
- Kept the original navy-blue colour palette for the Dark theme.
- Created a purpose-built Light theme with appropriate surfaces, borders, text colours, and contrast.
- Increased the contrast of the green **All question marks are ready** message in the Light theme for easier reading.
- Added a persistent theme preference that is remembered between visits.
- Replaced the original theme button with a compact Sun/Moon segmented control that shows the active theme within the toggle itself.
- Improved spacing between the workflow stages and theme switch.
- Simplified the header identity to **QIQ Paper Checker** with a quieter educator-focused subtitle.
- Reworked the Q logo from a generic purple treatment to a richer ink-blue identity that matches the educator workspace.
- Added a restrained cyan accent, inner highlight, stronger edge definition, and softer depth to keep the logo distinctive without overpowering the header.
- Constrained the desktop workspace to the viewport so the outer page and header remain fixed.
- Added independent internal scrolling to both the left setup panel and right review panel while preserving their complete bottom boundaries.
- Kept normal page scrolling on tablet and mobile layouts.
- Hid panel scrollbars while idle and revealed them with stronger contrast during mouse-wheel, touch, or direct scrolling.
- Automatically hid active scrollbars shortly after scrolling stops.

## Verification

- Rebuilt the application after the refinements.
- Re-ran the relevant identity, choice, marking, and pipeline tests.
- Checked the source changes for formatting errors.
