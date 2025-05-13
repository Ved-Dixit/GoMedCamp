// src/components/ViewCampReviewsPage.jsx
import React, { useState, useEffect, useCallback } from 'react';
import api from '../services/api';
import { Star } from 'lucide-react';
import '../App.css';

const ViewCampReviewsPage = ({ campId, currentUser, setCurrentPage }) => {
  const [reviews, setReviews] = useState([]);
  const [campName, setCampName] = useState(`Camp ID ${campId || 'N/A'}`);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const fetchCampDetailsAndReviews = useCallback(async () => {
    if (!campId || !currentUser?.id) {
      setError("Camp ID or User ID is missing.");
      setLoading(false);
      return;
    }
    setLoading(true);
    setError('');
    try {
      // Fetch camp name (optional, but good for display)
      const campDetailsRes = await api.get(`/organizer/camps/${campId}`);
      setCampName(campDetailsRes.data.name || `Camp ID ${campId}`);

      // Fetch reviews for this camp
      const reviewsRes = await api.get(`/camps/${campId}/reviews`);
      setReviews(reviewsRes.data || []);
    } catch (err) {
      setError('Failed to load camp reviews. ' + (err.response?.data?.error || err.message));
      console.error("Fetch camp reviews error:", err);
    } finally {
      setLoading(false);
    }
  }, [campId, currentUser]);

  useEffect(() => {
    if (!currentUser || currentUser.userType !== 'organizer') {
      setCurrentPage('login'); // Or 'home' or 'organizerDashboard'
      return;
    }
    fetchCampDetailsAndReviews();
  }, [currentUser, campId, setCurrentPage, fetchCampDetailsAndReviews]);

  if (loading) return <div className="page-loading-message">Loading reviews...</div>;

  return (
    <div className="container-page view-camp-reviews-page">
      <button
        onClick={() => setCurrentPage('campDetails')}
        className="back-button"
        style={{ marginBottom: '20px', backgroundColor: 'var(--secondary-button-bg)', color: 'var(--secondary-button-text)', border: '1px solid var(--secondary-button-border)' }}
      >
        &larr; Back to Camp Details
      </button>
      <div className="page-header">
        <h1>Reviews for {campName}</h1>
      </div>

      {error && <div className="error-message">{error}</div>}

      {!error && reviews.length === 0 && !loading && (
        <p className="info-message">No reviews have been submitted for this camp yet.</p>
      )}

      {!error && reviews.length > 0 && (
        <ul className="reviews-list">
          {reviews.map(review => (
            <li key={review.id} className="review-item">
              <div className="review-header">
                <strong>Patient: {review.patient_name || `User ID ${review.patient_user_id}`}</strong>
                <div className="review-rating">
                  {[...Array(5)].map((_, i) => (
                    <Star key={i} size={18} className={i < review.rating ? 'filled' : 'empty'} />
                  ))}
                </div>
              </div>
              {review.comment && <p className="review-comment">{review.comment}</p>}
              <p className="review-date">
                Submitted on: {new Date(review.created_at).toLocaleDateString()}
              </p>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default ViewCampReviewsPage;
