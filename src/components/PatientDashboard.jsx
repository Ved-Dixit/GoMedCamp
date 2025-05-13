// src/components/PatientDashboard.jsx
import React, { useState, useEffect, useCallback, useRef } from 'react';
import api from '../services/api';
import '../App.css';
import { Mic, MicOff, Volume2, AlertTriangle, Languages, Star } from 'lucide-react'; // Added Star

// MyPatientRecords component (remains the same from your previous version)
const MyPatientRecords = ({ currentUser }) => {
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const fetchMyRecords = useCallback(async () => {
    if (!currentUser?.id) {
      setError("User not identified. Cannot fetch records.");
      setLoading(false);
      return;
    }
    setLoading(true);
    setError('');
    try {
      const response = await api.get('/patient/my-details');
      setRecords(response.data || []);
    } catch (err) {
      if (err.response && err.response.status === 404) {
        setRecords([]);
        setError('No patient records found associated with your account. Organizers can add your details after a camp visit.');
      } else {
        setError('Failed to load your patient records. ' + (err.response?.data?.error || err.message));
        console.error("Fetch patient records error:", err);
      }
    } finally {
      setLoading(false);
    }
  }, [currentUser]);

  useEffect(() => {
    fetchMyRecords();
  }, [fetchMyRecords]);

  if (loading) {
    return <div className="patient-dashboard-section"><p>Loading your records...</p></div>;
  }

  return (
    <div className="patient-dashboard-section my-records-section">
      <h3>Your Camp Visit Records</h3>
      {error && records.length === 0 && <p className="info-message">{error}</p>}
      {error && records.length > 0 && <p className="error-message">{error}</p>}
      
      {records.length > 0 ? (
        <ul className="records-list">
          {records.map(record => (
            <li key={record.id} className="record-item">
              <h4>Camp: {record.camp_name || `ID ${record.camp_id}`}</h4>
              <p><strong>Your Name in Record:</strong> {record.name}</p>
              <p><strong>Email in Record:</strong> {record.email}</p>
              {record.phone_number && <p><strong>Phone:</strong> {record.phone_number}</p>}
              {record.disease_detected && <p><strong>Condition/Notes by Organizer:</strong> {record.disease_detected}</p>}
              {record.area_location && <p><strong>Area:</strong> {record.area_location}</p>}
              {record.organizer_notes && <p><strong>Additional Organizer Notes:</strong> {record.organizer_notes}</p>}
              <p><small>Record Date: {new Date(record.created_at).toLocaleDateString()}</small></p>
            </li>
          ))}
        </ul>
      ) : (
        !error && <p>You have no patient records from camps yet. If an organizer added you after a camp visit, your details might appear here.</p>
      )}
      <p style={{marginTop: '20px', fontSize: '0.9em', color: 'var(--text-color-muted)'}}>
        Note: These records are based on information provided by camp organizers.
        The AI Chatbot (below) can use your latest record for context.
      </p>
    </div>
  );
};


// VoiceBotInteraction component (remains the same from your previous version)
const supportedLanguages = [
  { code: 'en', name: 'English', speechApiLang: 'en-US' },
  { code: 'hi', name: 'Hindi (हिन्दी)', speechApiLang: 'hi-IN' },
  { code: 'es', name: 'Spanish (Español)', speechApiLang: 'es-ES' },
  { code: 'fr', name: 'French (Français)', speechApiLang: 'fr-FR' },
  { code: 'bn', name: 'Bengali (বাংলা)', speechApiLang: 'bn-BD' },
  { code: 'te', name: 'Telugu (తెలుగు)', speechApiLang: 'te-IN' },
  { code: 'mr', name: 'Marathi (मराठी)', speechApiLang: 'mr-IN' },
  { code: 'ta', name: 'Tamil (தமிழ்)', speechApiLang: 'ta-IN' },
];

const VoiceBotInteraction = ({ currentUser }) => {
  const [isListening, setIsListening] = useState(false);
  const [conversation, setConversation] = useState([]);
  const [statusMessage, setStatusMessage] = useState('Select language & click mic.');
  const [voiceBotError, setVoiceBotError] = useState('');
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [latestPatientRecordId, setLatestPatientRecordId] = useState(null);
  const [selectedLanguage, setSelectedLanguage] = useState(supportedLanguages[0]);

  const recognitionRef = useRef(null);
  const synthesisRef = useRef(window.speechSynthesis);

  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setVoiceBotError("Speech recognition is not supported by your browser.");
      return;
    }
    if (!synthesisRef.current) {
      setVoiceBotError("Speech synthesis is not supported by your browser.");
      return;
    }

    recognitionRef.current = new SpeechRecognition();
    recognitionRef.current.continuous = false;
    recognitionRef.current.interimResults = false;
    recognitionRef.current.lang = selectedLanguage.speechApiLang;

    recognitionRef.current.onstart = () => {
      setIsListening(true);
      setStatusMessage(`Listening in ${selectedLanguage.name}...`);
      setVoiceBotError('');
    };

    recognitionRef.current.onresult = async (event) => {
      const transcript = event.results[0][0].transcript;
      setConversation(prev => [...prev, { type: 'user', text: transcript, lang: selectedLanguage.code }]);
      setStatusMessage(`You said: "${transcript}". Processing...`);
      processUserInput(transcript);
    };

    recognitionRef.current.onerror = (event) => {
      setIsListening(false);
      let errorMsg = 'Speech recognition error: ' + event.error;
      if (event.error === 'no-speech') errorMsg = 'No speech detected. Please try again.';
      else if (event.error === 'audio-capture') errorMsg = 'Microphone problem. Please check your microphone.';
      else if (event.error === 'not-allowed') errorMsg = 'Permission to use microphone was denied.';
      else if (event.error === 'language-not-supported') errorMsg = `Speech recognition for ${selectedLanguage.name} may not be fully supported.`;
      setVoiceBotError(errorMsg);
      setStatusMessage('Error. Click mic to try again.');
    };

    recognitionRef.current.onend = () => {
      setIsListening(false);
      if (!voiceBotError && statusMessage.startsWith('Listening')) {
        setStatusMessage('Click the mic to start talking.');
      }
    };

    const fetchLatestRecord = async () => {
      if (!currentUser?.id) return;
      try {
        const response = await api.get('/patient/my-details');
        if (response.data && response.data.length > 0) {
          setLatestPatientRecordId(response.data[0].id);
        }
      } catch (err) {
        console.warn("Could not fetch patient record for bot context:", err);
      }
    };
    fetchLatestRecord();

    return () => {
      recognitionRef.current?.abort();
      synthesisRef.current?.cancel();
    };
  }, [currentUser, selectedLanguage, voiceBotError, statusMessage]);

  const translateText = async (text, targetLang, sourceLang = 'auto') => {
    if (!text || targetLang === sourceLang) return text;
    try {
      const response = await api.post('/translate', { text, target_lang: targetLang, source_lang: sourceLang });
      return response.data.translated_text;
    } catch (err) {
      console.error("Translation error:", err);
      setVoiceBotError(`Translation failed: ${err.message}. Using original text.`);
      return text;
    }
  };

  const processUserInput = async (userInput) => {
    let textToSendToBot = userInput;
    if (selectedLanguage.code !== 'en') {
      setStatusMessage(`Translating your message from ${selectedLanguage.name} to English...`);
      textToSendToBot = await translateText(userInput, 'en', selectedLanguage.code);
    }

    try {
      const response = await api.post('/patient/chatbot', {
        message: textToSendToBot,
        language: selectedLanguage.code,
        patient_record_id: latestPatientRecordId,
      });
      let botReply = response.data.reply;

      if (selectedLanguage.code !== 'en') {
        setStatusMessage(`Translating bot's response to ${selectedLanguage.name}...`);
        botReply = await translateText(botReply, selectedLanguage.code, 'en');
      }
      setConversation(prev => [...prev, { type: 'bot', text: botReply, lang: selectedLanguage.code }]);
      speakText(botReply, selectedLanguage.speechApiLang);
    } catch (err) {
      const errorMsg = 'Error getting response from bot: ' + (err.response?.data?.error || err.message);
      setVoiceBotError(errorMsg);
      setConversation(prev => [...prev, { type: 'bot', text: "Sorry, I couldn't process that.", lang: selectedLanguage.code }]);
      setStatusMessage('Error. Click mic to try again.');
    }
  };

  const speakText = (text, lang) => {
    if (!synthesisRef.current || !text) return;
    synthesisRef.current.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = lang;
    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => {
      setIsSpeaking(false);
      setStatusMessage('Click the mic to talk or type your message.');
    };
    utterance.onerror = (event) => {
      setIsSpeaking(false);
      setVoiceBotError(`Speech synthesis error for ${lang}: ${event.error}.`);
    };
    synthesisRef.current.speak(utterance);
  };

  const handleMicClick = () => {
    if (voiceBotError && voiceBotError.includes("not supported")) return;
    if (isListening) recognitionRef.current?.stop();
    else if (isSpeaking) {
      synthesisRef.current?.cancel();
      setIsSpeaking(false);
    } else {
      try {
        recognitionRef.current.lang = selectedLanguage.speechApiLang;
        recognitionRef.current?.start();
      } catch (e) {
        setVoiceBotError("Could not start listening. " + e.message);
      }
    }
  };
  
  const [manualInput, setManualInput] = useState('');
  const handleSendManualInput = async (e) => {
    e.preventDefault();
    if (!manualInput.trim()) return;
    const userText = manualInput.trim();
    setConversation(prev => [...prev, { type: 'user', text: userText, lang: selectedLanguage.code }]);
    setStatusMessage(`You typed: "${userText}". Processing...`);
    setManualInput('');
    processUserInput(userText);
  };

  const handleLanguageChange = (e) => {
    const newLangCode = e.target.value;
    const langObj = supportedLanguages.find(l => l.code === newLangCode);
    if (langObj) {
      setSelectedLanguage(langObj);
      setStatusMessage(`Language set to ${langObj.name}. Click mic or type.`);
      setVoiceBotError('');
      if (recognitionRef.current) {
        recognitionRef.current.lang = langObj.speechApiLang;
      }
    }
  };

  return (
    <div className="patient-dashboard-section voice-bot-section">
      <div className="voice-bot-header">
        <h3>AI Voice Assistant</h3>
        <div className="language-selector-container">
          <Languages size={18} style={{ marginRight: '5px' }} />
          <select value={selectedLanguage.code} onChange={handleLanguageChange} aria-label="Select language">
            {supportedLanguages.map(lang => (
              <option key={lang.code} value={lang.code}>{lang.name}</option>
            ))}
          </select>
        </div>
      </div>
      {voiceBotError && (
        <p className="error-message voice-bot-error">
          <AlertTriangle size={18} style={{ marginRight: '8px', verticalAlign: 'middle' }} />
          {voiceBotError}
        </p>
      )}
      <div className="voice-bot-controls">
        <button
          onClick={handleMicClick}
          className={`mic-button ${isListening ? 'listening' : ''} ${isSpeaking ? 'speaking' : ''}`}
          title={isListening ? "Stop Listening" : isSpeaking ? "Stop Speaking" : "Start Listening"}
          disabled={voiceBotError.includes("not supported by your browser")}
        >
          {isListening ? <MicOff size={24} /> : <Mic size={24} />}
        </button>
        <p className="status-message">{statusMessage}</p>
      </div>
      <div className="conversation-area">
        {conversation.map((entry, index) => (
          <div key={index} className={`message-bubble-voicebot ${entry.type}`}>
            <span className="sender-label">{entry.type === 'user' ? 'You' : 'Bot'} ({entry.lang})</span>
            <p>{entry.text}</p>
          </div>
        ))}
        {isSpeaking && conversation[conversation.length -1]?.type === 'bot' && (
            <Volume2 size={18} className="speaking-indicator" />
        )}
      </div>
      <form onSubmit={handleSendManualInput} className="manual-input-form">
        <input 
            type="text"
            value={manualInput}
            onChange={(e) => setManualInput(e.target.value)}
            placeholder="Or type your message here..."
            disabled={isListening || isSpeaking}
        />
        <button type="submit" disabled={isListening || isSpeaking || !manualInput.trim()}>Send</button>
      </form>
    </div>
  );
};

// --- NEW Review Submission Component ---
const ReviewSubmission = ({ currentUser }) => {
  const [camps, setCamps] = useState([]);
  const [selectedCampId, setSelectedCampId] = useState('');
  const [rating, setRating] = useState(0);
  const [hoverRating, setHoverRating] = useState(0);
  const [comment, setComment] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingCamps, setLoadingCamps] = useState(true);

  useEffect(() => {
    const fetchCamps = async () => {
      setLoadingCamps(true);
      try {
        const response = await api.get('/camps'); // Endpoint to list camps for review
        setCamps(response.data || []);
      } catch (err) {
        setError('Failed to load camps for review. ' + (err.response?.data?.error || err.message));
      } finally {
        setLoadingCamps(false);
      }
    };
    fetchCamps();
  }, []);

  const handleSubmitReview = async (e) => {
    e.preventDefault();
    if (!selectedCampId) {
      setError('Please select a camp to review.');
      return;
    }
    if (rating === 0) {
      setError('Please provide a star rating.');
      return;
    }
    setLoading(true);
    setError('');
    setSuccess('');
    try {
      await api.post('/reviews', {
        campId: parseInt(selectedCampId),
        rating: rating,
        comment: comment,
      });
      setSuccess('Review submitted successfully! Thank you for your feedback.');
      setSelectedCampId('');
      setRating(0);
      setComment('');
    } catch (err) {
      setError('Failed to submit review. ' + (err.response?.data?.error || err.message));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="patient-dashboard-section review-submission-section">
      <h3>Submit a Camp Review</h3>
      {error && <p className="error-message">{error}</p>}
      {success && <p className="success-message">{success}</p>}
      <form onSubmit={handleSubmitReview} className="review-form">
        <div className="form-group">
          <label htmlFor="campSelect">Select Camp:</label>
          {loadingCamps ? <p>Loading camps...</p> : (
            <select 
              id="campSelect" 
              value={selectedCampId} 
              onChange={(e) => setSelectedCampId(e.target.value)}
              required
            >
              <option value="">-- Select a Camp --</option>
              {camps.map(camp => (
                <option key={camp.id} value={camp.id}>{camp.name}</option>
              ))}
            </select>
          )}
        </div>
        <div className="form-group">
          <label>Your Rating:</label>
          <div className="star-rating">
            {[1, 2, 3, 4, 5].map(star => (
              <Star
                key={star}
                className={`star-icon ${(hoverRating || rating) >= star ? 'filled' : ''}`}
                onClick={() => setRating(star)}
                onMouseEnter={() => setHoverRating(star)}
                onMouseLeave={() => setHoverRating(0)}
                size={28}
              />
            ))}
          </div>
        </div>
        <div className="form-group">
          <label htmlFor="comment">Comments (Optional):</label>
          <textarea
            id="comment"
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            rows="4"
            placeholder="Share your experience..."
          />
        </div>
        <button type="submit" className="button-primary" disabled={loading || loadingCamps}>
          {loading ? 'Submitting...' : 'Submit Review'}
        </button>
      </form>
    </div>
  );
};


const PatientDashboard = ({ currentUser, setCurrentPage }) => {
  if (!currentUser || currentUser.userType !== 'requester') {
    setCurrentPage('login');
    return null;
  }

  return (
    <div className="container-page patient-dashboard-page">
      <div className="page-header">
        <h1>Patient Dashboard</h1>
        <p>Welcome, {currentUser.username || currentUser.email}! View your camp-related information and use the AI assistant.</p>
      </div>
      
      <MyPatientRecords currentUser={currentUser} />
      <hr className="section-divider" />
      <ReviewSubmission currentUser={currentUser} /> {/* ADDED Review Submission Section */}
      <hr className="section-divider" />
      <VoiceBotInteraction currentUser={currentUser} />
    </div>
  );
};

export default PatientDashboard;
