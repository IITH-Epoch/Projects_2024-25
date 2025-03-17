let globalOrigWords = [];
let globalCorrWords = [];
let activeSuggestionIndex = -1;

document.addEventListener("DOMContentLoaded", function () {
  const sentenceInput = document.getElementById("sentenceInput");
  const wordCountElement = document.getElementById("wordCount");
  const charCountElement = document.getElementById("charCount");
  const totalWordCountElement = document.getElementById("totalWordCount");
  const sentenceCountElement = document.getElementById("sentenceCount");

  // Initialize the word and character count
  updateWordAndCharCount(sentenceInput.value);

  sentenceInput.addEventListener("input", function () {
    const text = sentenceInput.value;
    
    // Update word and character count
    updateWordAndCharCount(text);
    
    const trimmed = text.trim();
    if (trimmed.slice(-1) === '.' || trimmed.slice(-1) === '?') {
      getLiveSuggestion(text);
    } else {
      clearRenderedText();
      activeSuggestionIndex = -1;
    }
  });

  // Function to update word and character count
  function updateWordAndCharCount(text) {
    // Count words
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;
    wordCountElement.textContent = words;
    totalWordCountElement.textContent = words;

    // Count characters
    const chars = text.length;
    charCountElement.textContent = chars;

    // Count sentences
    const sentences = text.trim() ? (text.match(/[^.?!]+[.?!]+/g) || []).length : 0;
    sentenceCountElement.textContent = sentences;

    // Update accuracy rate (just a placeholder since we don't have real data)
    const correctionCount = document.getElementById("correctionCount");
    const accuracyRate = document.getElementById("accuracyRate");
    
    // Set some default values or calculate based on actual corrections
    if (words > 0) {
      const corrections = parseInt(correctionCount.textContent) || 0;
      const accuracy = words > 0 ? Math.round(((words - corrections) / words) * 100) : 100;
      accuracyRate.textContent = `${accuracy}%`;
    }
  }
});

// Listen for keyboard events globally to allow cycling through suggestions
document.addEventListener("keydown", function (e) {
  const container = document.getElementById("renderedText");
  if (!container) return;
  const suggestions = container.querySelectorAll(".highlight");
  if (suggestions.length === 0) return;
  
  if (e.ctrlKey && e.key === "]") {
    e.preventDefault();
    activeSuggestionIndex = (activeSuggestionIndex + 1) % suggestions.length;
    updateActiveSuggestionHighlight(suggestions);
  } else if (e.ctrlKey && e.key === "[") {
    e.preventDefault();
    activeSuggestionIndex = (activeSuggestionIndex - 1 + suggestions.length) % suggestions.length;
    updateActiveSuggestionHighlight(suggestions);
  } else if (e.key === "Enter") {
    if (activeSuggestionIndex !== -1) {
      e.preventDefault();
      const activeSuggestion = suggestions[activeSuggestionIndex];
      if (activeSuggestion) {
        const dataIndex = activeSuggestion.getAttribute("data-index");
        if (dataIndex !== null) {
          const index = parseInt(dataIndex);
          console.log("Keyboard: Accepting word suggestion at index", index);
          acceptWordSuggestion(index);
        } else {
          const corrected = activeSuggestion.getAttribute("data-correction");
          console.log("Keyboard: Accepting sentence suggestion:", corrected);
          acceptSentenceSuggestion(corrected);
        }
        activeSuggestionIndex = -1;
        updateActiveSuggestionHighlight(suggestions);
      }
    }
  }
});

// Helper to update the visual style of active suggestion items
function updateActiveSuggestionHighlight(suggestions) {
  suggestions.forEach((el, idx) => {
    if (idx === activeSuggestionIndex) {
      el.classList.add("active-suggestion");
    } else {
      el.classList.remove("active-suggestion");
    }
  });
}

async function getLiveSuggestion(text) {
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text })
    });
    if (!response.ok) {
      console.error("Error: " + response.statusText);
      return;
    }
    const data = await response.json();
    const correctedLastSentence = data.corrected_last_sentence;
    const originalLastSentence = extractLastSentence(text);

    globalOrigWords = originalLastSentence.split(/\s+/).filter(w => w.length > 0);
    globalCorrWords = correctedLastSentence.split(/\s+/).filter(w => w.length > 0);

    activeSuggestionIndex = -1;

    // Update corrections count in the stats
    const correctionCount = document.getElementById("correctionCount");
    let corrections = 0;
    
    // Count differences between original and corrected words
    for (let i = 0; i < Math.min(globalOrigWords.length, globalCorrWords.length); i++) {
      if (globalOrigWords[i] !== globalCorrWords[i]) {
        corrections++;
      }
    }
    
    // Add difference in length if sentences have different number of words
    corrections += Math.abs(globalOrigWords.length - globalCorrWords.length);
    
    correctionCount.textContent = corrections;
    
    // Update accuracy rate
    const totalWords = parseInt(document.getElementById("totalWordCount").textContent);
    const accuracyRate = document.getElementById("accuracyRate");
    const accuracy = totalWords > 0 ? Math.round(((totalWords - corrections) / totalWords) * 100) : 100;
    accuracyRate.textContent = `${accuracy}%`;

    if (globalOrigWords.length === globalCorrWords.length) {
      renderWordDiff(globalOrigWords, globalCorrWords);
    } else {
      renderSentenceDiff(originalLastSentence, correctedLastSentence);
    }
  } catch (error) {
    console.error("Error: " + error.message);
    
    // For demonstration/testing purposes only:
    // Mock response if server is not available
    mockLiveSuggestion(text);
  }
}

// Mock function for demonstration when server is not available
function mockLiveSuggestion(text) {
  const originalLastSentence = extractLastSentence(text);
  
  // Simple mock corrections for demonstration
  let correctedLastSentence = originalLastSentence;
  
  // Sample corrections
  correctedLastSentence = correctedLastSentence.replace(" teh ", " the ");
  correctedLastSentence = correctedLastSentence.replace(" thier ", " their ");
  correctedLastSentence = correctedLastSentence.replace(" recieve ", " receive ");
  
  globalOrigWords = originalLastSentence.split(/\s+/).filter(w => w.length > 0);
  globalCorrWords = correctedLastSentence.split(/\s+/).filter(w => w.length > 0);
  
  activeSuggestionIndex = -1;
  
  if (globalOrigWords.length === globalCorrWords.length) {
    renderWordDiff(globalOrigWords, globalCorrWords);
  } else {
    renderSentenceDiff(originalLastSentence, correctedLastSentence);
  }
}

function extractLastSentence(text) {
  // Capture sentences ending with . or ? using regex
  const sentences = text.match(/[^.?!]+[.?!]+/g);
  if (sentences && sentences.length > 0) {
    return sentences[sentences.length - 1].trim();
  } else {
    return text.trim();
  }
}

function renderWordDiff(origWords, corrWords) {
  const container = document.getElementById("renderedText");
  container.innerHTML = "";
  
  // Render each word as a span with a data-index attribute.
  origWords.forEach((word, i) => {
    const span = document.createElement("span");
    span.innerText = word + " ";
    span.setAttribute("data-index", i);
    span.style.position = "relative"; 
    span.style.cursor = "pointer";    
    if (word !== corrWords[i]) {
      span.className = "highlight";
      span.setAttribute("data-correction", corrWords[i]);
      const tooltip = document.createElement("div");
      tooltip.className = "tooltip";
      tooltip.innerText = corrWords[i];
      tooltip.style.pointerEvents = "none";
      
      span.appendChild(tooltip);
    }
    container.appendChild(span);
  });
  
  // Attach a single click event listener to the container using event delegation.
  container.addEventListener("click", function (e) {
    let target = e.target;
    while (target && target !== container) {
      if (target.hasAttribute("data-index")) {
        const index = parseInt(target.getAttribute("data-index"));
        acceptWordSuggestion(index);
        break;
      }
      target = target.parentNode;
    }
  });
}

function renderSentenceDiff(original, corrected) {
  const container = document.getElementById("renderedText");
  container.innerHTML = "";
  const span = document.createElement("span");
  span.className = "highlight";
  span.innerText = original;
  span.setAttribute("data-correction", corrected);
  span.style.position = "relative";
  const tooltip = document.createElement("div");
  tooltip.className = "tooltip";
  tooltip.innerText = corrected;
  span.appendChild(tooltip);

  // On click, accept full sentence correction
  span.addEventListener("click", function () {
    acceptSentenceSuggestion(corrected);
  });
  container.appendChild(span);
}

function clearRenderedText() {
  document.getElementById("renderedText").innerHTML = "";
}

// Accept correction for a single word: update only that word in the textarea (typing bar)
function acceptWordSuggestion(wordIndex) {
  console.log("ENTERED acceptWordSuggestion");
  const sentenceInput = document.getElementById("sentenceInput");
  let text = sentenceInput.value;
  const sentences = text.match(/[^.?!]+[.?!]+/g);
  if (!sentences || sentences.length === 0) return;
  
  let lastSentence = sentences[sentences.length - 1].trim();
  let words = lastSentence.split(/\s+/);
  if (wordIndex < words.length) {
    words[wordIndex] = globalCorrWords[wordIndex];
    const newLastSentence = words.join(" ");
    sentences[sentences.length - 1] = newLastSentence;
    const newText = sentences.join(" ");
    sentenceInput.value = newText;
    globalOrigWords[wordIndex] = globalCorrWords[wordIndex];
    // Re-run suggestion update to refresh diff
    getLiveSuggestion(newText);
  }
}

// Accept full sentence correction
function acceptSentenceSuggestion(correctedSentence) {
  const sentenceInput = document.getElementById("sentenceInput");
  let text = sentenceInput.value;
  const sentences = text.match(/[^.?!]+[.?!]+/g);
  if (!sentences || sentences.length === 0) return;
  sentences[sentences.length - 1] = correctedSentence;
  const newText = sentences.join(" ");
  sentenceInput.value = newText;
  globalOrigWords = correctedSentence.split(/\s+/).filter(w => w.length > 0);
  globalCorrWords = [...globalOrigWords];
}

// Create and show a notification
function showNotification(message) {
  // Remove any existing notification
  const existingNotification = document.querySelector('.notification');
  if (existingNotification) {
    document.body.removeChild(existingNotification);
  }
  
  // Create new notification
  const notification = document.createElement('div');
  notification.className = 'notification';
  notification.textContent = message;
  document.body.appendChild(notification);
  
  // Trigger animation
  setTimeout(() => {
    notification.classList.add('show');
  }, 10);
  
  // Auto-hide after 3 seconds
  setTimeout(() => {
    notification.classList.remove('show');
    setTimeout(() => {
      if (notification.parentNode) {
        document.body.removeChild(notification);
      }
    }, 300); // Wait for transition to complete
  }, 3000);
}