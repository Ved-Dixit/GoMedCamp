// src/components/Footer.jsx
import React from 'react';
import { Linkedin, Instagram, Youtube, Twitter, Facebook } from 'lucide-react'; // Added Twitter and Facebook as examples

const Footer = () => {
  // Placeholder URLs for social media - replace with actual links
  const socialLinks = {
    linkedin: "https://www.linkedin.com/in/ved-dixit-2280b2324/",
    instagram: "https://www.instagram.com/?next=%2F",
    youtube: "https://www.youtube.com/@veddixit4246",
  };


  return (
    <footer className="footer">
      <div className="footer-container"> {/* For max-width and centering */}
        <div className="footer-main-content">
          <div className="footer-column">
            <h4>GoMedCamp</h4>
          </div>

          <div className="footer-column">
            <h4>Resources</h4>
            <ul>
              <li><a href={socialLinks.youtube} target="_blank" rel="noopener noreferrer">Watch Demo</a></li>
            </ul>
          </div>
          
          <div className="footer-column footer-column-connect">
            <h4>Connect With Us</h4>
            <div className="footer-social-icons">
              <a href={socialLinks.linkedin} target="_blank" rel="noopener noreferrer" aria-label="LinkedIn">
                <Linkedin size={24} />
              </a>
              <a href={socialLinks.instagram} target="_blank" rel="noopener noreferrer" aria-label="Instagram">
                <Instagram size={24} />
              </a>
              <a href={socialLinks.youtube} target="_blank" rel="noopener noreferrer" aria-label="YouTube">
                <Youtube size={24} />
              </a>
            </div>
          </div>
        </div>

        <div className="footer-bottom">
          <p>&copy; {new Date().getFullYear()} GoMedCamp. All rights reserved.</p>
          <p>Designed to make healthcare accessible.</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
