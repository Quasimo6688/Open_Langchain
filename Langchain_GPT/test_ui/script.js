document.getElementById('chat-form').addEventListener('submit', function(e) {
    e.preventDefault();

    const name = this.querySelector('input[type="text"]').value;
    const userId = this.querySelector('input[type="text"]:nth-child(2)').value;
    const message = this.querySelector('textarea').value;

    // Send message to the backend
    fetch(`/dialogue/?message=${encodeURIComponent(message)}&session_id=${encodeURIComponent(userId)}`, {
        method: 'GET'
    }).then(response => {
        response.text().then(data => {
            // Display response in the messages area
            const messagesArea = document.querySelector('.messages');
            messagesArea.innerHTML += `<p>${data}</p>`;
        });
    });

    // Clear input fields
    this.querySelector('textarea').value = '';
});

// SSE listener for backend messages
const evtSource = new EventSource("/dialogue/");
evtSource.onmessage = function(event) {
    const messageData = JSON.parse(event.data);
    // Display received message
    const messagesArea = document.querySelector('.messages');
    messagesArea.innerHTML += `<p>${messageData}</p>`;
};
