// src/main.jsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx';
import './index.css'; // Vite's default global styles

// Import slick-carousel styles HERE
import 'slick-carousel/slick/slick.css';
import 'slick-carousel/slick/slick-theme.css';

// App.css is imported in App.jsx, which is fine.

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
