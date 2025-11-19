document.addEventListener("DOMContentLoaded", () => {
  const bubble = document.getElementById("chatBubble");
  const windowEl = document.getElementById("chatWindow");
  const closeBtn = document.getElementById("closeChat");
  const form = document.getElementById("chatForm");
  const input = document.getElementById("chatInput");
  const chatBody = document.getElementById("chatBody");
  const quickMenu = document.getElementById("chatQuickMenu");
  const closeQuickMenu = document.getElementById("closeQuickMenu");

  // KIỂM TRA TRẠNG THÁI LƯU (localStorage)
  const quickMenuHidden = localStorage.getItem("quickMenuHidden") === "true";

  if (quickMenuHidden && quickMenu) {
    quickMenu.style.display = "none";
    showRestoreButton(); // Hiển thị nút "Hiện gợi ý lại"
  }

  // XỬ LÝ CHAT CƠ BẢN
  bubble.addEventListener("click", () => {
    windowEl.style.display = "flex";
    bubble.style.display = "none";
    loadChatHistory();
  });
  closeBtn.addEventListener("click", () => {
    windowEl.style.display = "none";
    bubble.style.display = "flex";
  });

  async function loadChatHistory() {
    try {
      const res = await fetch("/chat_ai_history");
      const data = await res.json();
      chatBody.innerHTML = "";
      if (data.messages && data.messages.length > 0) {
        data.messages.forEach((m) => {
          chatBody.insertAdjacentHTML(
            "beforeend",
            `<div class='msg-user'><div class='bubble'>${m.user}</div></div>
               <div class='msg-ai'><div class='bubble'>${m.ai}</div></div>`
          );
        });
      } else {
        chatBody.innerHTML =
          "<div class='text-center text-muted small mt-3'>👋 Xin chào! Tôi là <b>Trợ lý CVD-AI</b>. Hãy hỏi tôi về tim mạch nhé 💓</div>";
      }
      chatBody.scrollTop = chatBody.scrollHeight;
    } catch (e) {
      console.error("Lỗi tải lịch sử:", e);
    }
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    chatBody.insertAdjacentHTML(
      "beforeend",
      `<div class='msg-user'><div class='bubble'>${text}</div></div>`
    );
    input.value = "";
    chatBody.scrollTop = chatBody.scrollHeight;

    const typing = document.createElement("div");
    typing.className = "msg-ai";
    typing.innerHTML = "<div class='bubble'><i>AI đang trả lời...</i></div>";
    chatBody.appendChild(typing);
    chatBody.scrollTop = chatBody.scrollHeight;

    try {
      const res = await fetch("/chat_ai_api", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });
      const data = await res.json();
      typing.remove();
      chatBody.insertAdjacentHTML(
        "beforeend",
        `<div class='msg-ai'><div class='bubble'>${data.reply}</div></div>`
      );
      chatBody.scrollTop = chatBody.scrollHeight;
    } catch {
      typing.innerHTML =
        "<div class='bubble text-danger'>⚠️ Lỗi kết nối đến AI.</div>";
    }
  });

  // XỬ LÝ NÚT GỢI Ý NHANH
  document.querySelectorAll(".quick-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const text = btn.dataset.msg;
      if (windowEl.style.display === "none" || !windowEl.style.display) {
        windowEl.style.display = "flex";
        bubble.style.display = "none";
      }
      input.value = text;
      input.focus();
    });
  });

  // ẨN VĨNH VIỄN CÁC GỢI Ý
  if (closeQuickMenu) {
    closeQuickMenu.addEventListener("click", () => {
      quickMenu.style.display = "none";
      localStorage.setItem("quickMenuHidden", "true"); // lưu trạng thái
      showRestoreButton();
    });
  }

  // HIỂN THỊ LẠI GỢI Ý (nút nhỏ nổi)
  function showRestoreButton() {
    const existing = document.getElementById("restoreQuickMenu");
    if (existing) return; // tránh trùng

    const restoreBtn = document.createElement("button");
    restoreBtn.id = "restoreQuickMenu";
    restoreBtn.innerHTML = "📎 Hiện gợi ý lại";
    restoreBtn.className = "btn btn-light border position-fixed";
    restoreBtn.style.cssText =
      "bottom:100px; right:25px; z-index:2200; font-size:0.85rem; border-radius:20px; padding:4px 12px; box-shadow:0 2px 6px rgba(0,0,0,0.2);";

    restoreBtn.addEventListener("click", () => {
      quickMenu.style.display = "flex";
      localStorage.setItem("quickMenuHidden", "false");
      restoreBtn.remove();
    });

    document.body.appendChild(restoreBtn);
  }
});
