// Professional NeuroMind AI Chatbot JavaScript
document.addEventListener("DOMContentLoaded", function () {
  // ================= DOM ELEMENTS =================
  const messageInput = document.getElementById("message-input");
  const sendButton = document.getElementById("send-button");
  const voiceBtn = document.getElementById("voice-btn");
  const chatMessages = document.getElementById("chat-messages");
  const memoryInsights = document.getElementById("memory-insights");
  const themeToggle = document.getElementById("theme-toggle");
  const settingsBtn = document.getElementById("settings-btn");
  const exportBtn = document.getElementById("export-btn");
  const settingsModal = document.getElementById("settings-modal");
  const toastContainer = document.getElementById("toast-container");
  const charCount = document.getElementById("char-count");
  const messageCount = document.getElementById("message-count");
  const memoryCount = document.getElementById("memory-count");
  const responseTime = document.getElementById("response-time");
  const memoryHits = document.getElementById("memory-hits");
  const sessionDuration = document.getElementById("session-duration");
  const typingStatus = document.getElementById("typing-status");

  // ================= MARKDOWN CONFIG =================
  if (typeof marked !== "undefined") {
    marked.setOptions({
      gfm: true,
      breaks: true
    });
  }

  // ================= STATE =================
  let isTyping = false;
  let messageHistory = [];
  let historyIndex = -1;
  let totalMessages = 0;
  let totalMemories = 0;
  let sessionStartTime = Date.now();
  let currentTheme = localStorage.getItem("theme") || "light";

  let settings = {
    autoScroll: true,
    showTimestamps: true,
    soundNotifications: false,
    memoryLimit: 3,
  };

  loadSettings();
  initializeApp();

  // ================= INIT =================
  function initializeApp() {
    setTheme(currentTheme);
    setInterval(updateSessionDuration, 1000);

    sendButton.addEventListener("click", handleSendMessage);
    messageInput.addEventListener("keypress", handleKeyPress);
    messageInput.addEventListener("input", handleInputChange);
    themeToggle.addEventListener("click", toggleTheme);
    settingsBtn.addEventListener("click", openSettingsModal);
    exportBtn.addEventListener("click", exportChat);
    voiceBtn.addEventListener("click", handleVoiceInput);

    document.getElementById("auto-scroll").addEventListener("change", updateSetting);
    document.getElementById("show-timestamps").addEventListener("change", updateSetting);
    document.getElementById("sound-notifications").addEventListener("change", updateSetting);
    document.getElementById("memory-limit").addEventListener("change", updateSetting);

    document.querySelector(".modal-close").addEventListener("click", closeSettingsModal);

    document.querySelectorAll(".collapse-btn").forEach(btn => {
      btn.addEventListener("click", toggleSection);
    });
  }

  // ================= SEND MESSAGE =================
  function handleSendMessage() {
    const message = messageInput.value.trim();
    if (!message || isTyping) return;

    addMessage("user", message);
    messageHistory.push(message);
    historyIndex = -1;
    totalMessages++;
    updateMessageCount();

    messageInput.value = "";
    handleInputChange();
    showTypingIndicator();

    const startTime = Date.now();

    fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: message,
        memory_limit: settings.memoryLimit
      }),
    })
      .then(res => res.json())
      .then(data => {
        hideTypingIndicator();

        if (data.error) {
          addMessage("bot", data.error, "error");
          return;
        }

        addMessage("bot", data.response);

        updateMemoryInsights(data.episodic_hits || [], data.semantic_hits || []);
        updateAnalytics(
          data.processing_time || (Date.now() - startTime),
          data.memory_count || 0
        );

        if (data.context && typeof renderContext === "function") {
          renderContext(data.context);
        }
      })
      .catch(err => {
        hideTypingIndicator();
        console.error(err);
        addMessage("bot", "Network error. Please try again.", "error");
      });
  }

  // ================= CONTEXT VIEW =================
  function renderContext(context) {
    const el = document.getElementById("context-content");
    if (!el) return;

    try {
      if (context.note) {
        el.textContent = context.note;
        return;
      }

      el.textContent = `
USER PERSONA:
${(context.persona || []).map(p => "- " + p).join("\n")}

KNOWN FACTS:
${(context.knowledge || []).map(k => "- " + k).join("\n")}

KNOWN PROCESSES:
${(context.process || []).map(p => "- " + p).join("\n")}

EPISODIC CONTEXT:
${(context.episodic || []).map(e => "- " + e.user).join("\n")}

SHORT TERM:
${(context.short_term || []).map(m => `${m.role}: ${m.content}`).join("\n")}

----------------------------
FINAL PROMPT
----------------------------
${context.final_prompt || ""}
`;
    } catch (e) {
      console.error("Context render error:", e);
    }
  }

  // ================= UI HELPERS =================
  function addMessage(sender, content, type = "normal") {
    const msg = document.createElement("div");
    msg.className = `message ${sender}`;

    let renderedContent = content;

    // ✅ Markdown rendering ONLY for bot messages
    if (sender === "bot" && typeof marked !== "undefined") {
      renderedContent = marked.parse(content);
    }

    msg.innerHTML = `
      <div class="avatar">${sender === "user" ? "👤" : "🤖"}</div>
      <div class="content chat-message">${renderedContent}</div>
    `;

    chatMessages.appendChild(msg);
    if (settings.autoScroll) chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function showTypingIndicator() {
    isTyping = true;
    typingStatus.textContent = "AI is thinking...";
  }

  function hideTypingIndicator() {
    isTyping = false;
    typingStatus.textContent = "";
  }

  function updateMemoryInsights(e, s) {
    totalMemories = e.length + s.length;
    memoryCount.textContent = totalMemories;
  }

  function updateAnalytics(ms, hits) {
    responseTime.textContent = `${ms}ms`;
    memoryHits.textContent = hits;
  }

  function updateSessionDuration() {
    const t = Date.now() - sessionStartTime;
    sessionDuration.textContent =
      String(Math.floor(t / 60000)).padStart(2, "0") +
      ":" +
      String(Math.floor((t % 60000) / 1000)).padStart(2, "0");
  }

  function updateMessageCount() {
    messageCount.textContent = `${totalMessages} messages`;
  }

  function handleKeyPress(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  }

  function handleInputChange() {
    charCount.textContent = messageInput.value.length;
  }

  function toggleTheme() {
    currentTheme = currentTheme === "light" ? "dark" : "light";
    setTheme(currentTheme);
    localStorage.setItem("theme", currentTheme);
  }

  function setTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
  }

  function toggleSection(e) {
    document.getElementById(e.currentTarget.dataset.target)
      .classList.toggle("collapsed");
  }

  function updateSetting(e) {
    settings[e.target.id.replace("-", "")] =
      e.target.type === "checkbox" ? e.target.checked : e.target.value;
    localStorage.setItem("chatSettings", JSON.stringify(settings));
  }

  function loadSettings() {
    const saved = localStorage.getItem("chatSettings");
    if (saved) settings = { ...settings, ...JSON.parse(saved) };
  }

  function openSettingsModal() {
    settingsModal.classList.add("show");
  }

  function closeSettingsModal() {
    settingsModal.classList.remove("show");
  }

  function exportChat() {}
  function handleVoiceInput() {}
});
