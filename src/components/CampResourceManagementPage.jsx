// src/components/CampResourceManagementPage.jsx
import React, { useState, useEffect, useCallback } from 'react';

const CampResourceManagementPage = ({ campId, currentUser, setCurrentPage }) => {
  const [campDetails, setCampDetails] = useState(null); // For camp name, dates
  const [targetPatients, setTargetPatients] = useState(0);
  const [staffList, setStaffList] = useState([]);
  const [medicineList, setMedicineList] = useState([]);
  const [equipmentList, setEquipmentList] = useState([]);

  const [newStaff, setNewStaff] = useState({ name: '', role: '', origin: '', contact: '', notes: '' });
  const [newMedicine, setNewMedicine] = useState({ name: '', unit: '', quantityPerPatient: 0, notes: '' });
  const [newEquipment, setNewEquipment] = useState({ name: '', quantity: 0, notes: '' });

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [saveMessage, setSaveMessage] = useState('');

  const [estimatedCost, setEstimatedCost] = useState({ min: 0, max: 0 });

  const placeholderCosts = {
    staffPerDay: 50, // Placeholder cost per staff member per day
    medicinePerUnit: 2, // Placeholder average cost per unit of medicine (needs to be more granular in real app)
    equipmentPerItem: 75, // Placeholder average cost per equipment item
  };

  const calculateCampDurationDays = useCallback(() => {
    if (campDetails && campDetails.start_date && campDetails.end_date) {
      const startDate = new Date(campDetails.start_date);
      const endDate = new Date(campDetails.end_date);
      const diffTime = Math.abs(endDate - startDate);
      const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24)) + 1; // Inclusive of start and end day
      return diffDays > 0 ? diffDays : 1; // Minimum 1 day
    }
    return 1; // Default duration if dates are not available
  }, [campDetails]);

  const calculateEstimatedCost = useCallback(() => {
    const durationDays = calculateCampDurationDays();

    const staffCost = staffList.reduce((total) => total + (placeholderCosts.staffPerDay * durationDays), 0);

    const medicineCost = medicineList.reduce((total, med) => {
      const qty = parseFloat(med.quantityPerPatient) || 0;
      const numPatients = parseInt(targetPatients, 10) || 0;
      return total + (qty * numPatients * placeholderCosts.medicinePerUnit);
    }, 0);

    const equipmentCost = equipmentList.reduce((total, equip) => {
      const qty = parseInt(equip.quantity, 10) || 0;
      return total + (qty * placeholderCosts.equipmentPerItem);
    }, 0);

    const totalBaseCost = staffCost + medicineCost + equipmentCost;
    setEstimatedCost({
      min: Math.round(totalBaseCost * 0.85), // Example: 85% of base
      max: Math.round(totalBaseCost * 1.15), // Example: 115% of base
    });
  }, [staffList, medicineList, equipmentList, targetPatients, calculateCampDurationDays, placeholderCosts]);


  useEffect(() => {
    const fetchCampData = async () => {
      if (!campId || !currentUser?.id) {
        setError("Camp ID or User ID is missing.");
        setLoading(false);
        return;
      }
      setLoading(true);
      setError('');
      setSaveMessage('');
      try {
        // Fetch Camp Details (for name, dates)
        const campDetailsRes = await fetch(`https://gomedcamp-1.onrender.com/api/organizer/camps/${campId}`, {
          headers: { 'X-User-Id': currentUser.id.toString() },
        });
        if (!campDetailsRes.ok) throw new Error(`Failed to fetch camp details: ${campDetailsRes.statusText}`);
        const campData = await campDetailsRes.json();
        setCampDetails(campData);

        // Fetch Camp Resources
        const resourcesRes = await fetch(`https://gomedcamp-1.onrender.com/api/organizer/camp/${campId}/resources`, {
          headers: { 'X-User-Id': currentUser.id.toString() },
        });
        if (!resourcesRes.ok) throw new Error(`Failed to fetch resources: ${resourcesRes.statusText}`);
        const resourcesData = await resourcesRes.json();
        
        setTargetPatients(resourcesData.targetPatients || 0);
        setStaffList(resourcesData.staffList || []);
        setMedicineList(resourcesData.medicineList || []);
        setEquipmentList(resourcesData.equipmentList || []);

      } catch (err) {
        setError(err.message);
        console.error("Fetch error:", err);
      } finally {
        setLoading(false);
      }
    };
    fetchCampData();
  }, [campId, currentUser]);

  useEffect(() => {
    calculateEstimatedCost();
  }, [staffList, medicineList, equipmentList, targetPatients, calculateEstimatedCost]);


  const handleInputChange = (setter, field, value) => {
    setter(prev => ({ ...prev, [field]: value }));
  };

  const addItem = (type) => {
    if (type === 'staff' && newStaff.name) {
      setStaffList([...staffList, { ...newStaff, id: `temp-${Date.now()}` }]); // Temp ID for list key
      setNewStaff({ name: '', role: '', origin: '', contact: '', notes: '' });
    } else if (type === 'medicine' && newMedicine.name) {
      setMedicineList([...medicineList, { ...newMedicine, id: `temp-${Date.now()}` }]);
      setNewMedicine({ name: '', unit: '', quantityPerPatient: 0, notes: '' });
    } else if (type === 'equipment' && newEquipment.name) {
      setEquipmentList([...equipmentList, { ...newEquipment, id: `temp-${Date.now()}` }]);
      setNewEquipment({ name: '', quantity: 0, notes: '' });
    }
  };

  const removeItem = (type, index) => {
    if (type === 'staff') setStaffList(staffList.filter((_, i) => i !== index));
    else if (type === 'medicine') setMedicineList(medicineList.filter((_, i) => i !== index));
    else if (type === 'equipment') setEquipmentList(equipmentList.filter((_, i) => i !== index));
  };
  
  // Note: Full edit functionality for list items would require more state/modals.
  // This example focuses on add/remove and saving the whole list.

  const handleSaveResources = async () => {
    if (!currentUser?.id) {
      setError("User not authenticated to save.");
      return;
    }
    setLoading(true);
    setError('');
    setSaveMessage('');
    try {
      const payload = {
        targetPatients: parseInt(targetPatients, 10) || 0,
        staffList: staffList.map(({ id, ...staff }) => staff), // Remove temp IDs
        medicineList: medicineList.map(({ id, ...med }) => ({
            ...med,
            quantityPerPatient: parseFloat(med.quantityPerPatient) || 0,
        })),
        equipmentList: equipmentList.map(({ id, ...equip }) => ({
            ...equip,
            quantity: parseInt(equip.quantity, 10) || 0,
        })),
      };

      const response = await fetch(`https://camp-mdxq.onrender.com/api/organizer/camp/${campId}/resources`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-User-Id': currentUser.id.toString(),
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.error || `Failed to save resources: ${response.statusText}`);
      }
      setSaveMessage("Resources saved successfully!");
      // Optionally re-fetch data or update IDs if backend returns them
    } catch (err) {
      setError(err.message);
      console.error("Save error:", err);
    } finally {
      setLoading(false);
      setTimeout(() => setSaveMessage(''), 3000);
    }
  };

  if (loading && !campDetails) return <p className="crm-loading">Loading camp resources...</p>;
  if (error) return <p className="crm-error-message">{error}</p>;

  return (
    <div className="camp-resource-management-page">
      <div className="crm-header">
        <h1>Resource Management for {campDetails?.name || `Camp #${campId}`}</h1>
        <div className="crm-summary-panel">
            <div className="crm-target-patients">
            <label htmlFor="targetPatients">Target Patients: </label>
            <input
                type="number"
                id="targetPatients"
                value={targetPatients}
                onChange={(e) => setTargetPatients(e.target.value)}
                min="0"
            />
            </div>
            <div className="crm-estimated-cost">
            <strong>Estimated Cost Range: </strong>
            <span>₹{estimatedCost.min.toLocaleString()} - ₹{estimatedCost.max.toLocaleString()}</span>
            </div>
        </div>
      </div>

      <div className="crm-sections-container">
        {/* Staff Management Section */}
        <section className="crm-section crm-staff">
          <h2>Staff Management</h2>
          <div className="crm-form">
            <input type="text" placeholder="Staff Name" value={newStaff.name} onChange={(e) => handleInputChange(setNewStaff, 'name', e.target.value)} />
            <input type="text" placeholder="Role" value={newStaff.role} onChange={(e) => handleInputChange(setNewStaff, 'role', e.target.value)} />
            <input type="text" placeholder="Origin (e.g., Hospital Name)" value={newStaff.origin} onChange={(e) => handleInputChange(setNewStaff, 'origin', e.target.value)} />
            <input type="text" placeholder="Contact" value={newStaff.contact} onChange={(e) => handleInputChange(setNewStaff, 'contact', e.target.value)} />
            <textarea placeholder="Notes" value={newStaff.notes} onChange={(e) => handleInputChange(setNewStaff, 'notes', e.target.value)} />
            <button onClick={() => addItem('staff')} disabled={loading}>Add Staff</button>
          </div>
          <ul className="crm-list">
            {staffList.map((staff, index) => (
              <li key={staff.id || index}>
                {staff.name} ({staff.role}) - {staff.contact}
                <button onClick={() => removeItem('staff', index)} className="crm-remove-btn" disabled={loading}>Remove</button>
              </li>
            ))}
          </ul>
        </section>

        {/* Medicine Management Section */}
        <section className="crm-section crm-medicines">
          <h2>Medicine Management</h2>
          <div className="crm-form">
            <input type="text" placeholder="Medicine Name" value={newMedicine.name} onChange={(e) => handleInputChange(setNewMedicine, 'name', e.target.value)} />
            <input type="text" placeholder="Unit (e.g., tablet, ml)" value={newMedicine.unit} onChange={(e) => handleInputChange(setNewMedicine, 'unit', e.target.value)} />
            <input type="number" placeholder="Qty per Patient" value={newMedicine.quantityPerPatient} onChange={(e) => handleInputChange(setNewMedicine, 'quantityPerPatient', e.target.value)} min="0" step="0.01"/>
            <textarea placeholder="Notes" value={newMedicine.notes} onChange={(e) => handleInputChange(setNewMedicine, 'notes', e.target.value)} />
            <button onClick={() => addItem('medicine')} disabled={loading}>Add Medicine</button>
          </div>
          <ul className="crm-list">
            {medicineList.map((med, index) => (
              <li key={med.id || index}>
                {med.name} ({med.quantityPerPatient} {med.unit}/patient)
                <button onClick={() => removeItem('medicine', index)} className="crm-remove-btn" disabled={loading}>Remove</button>
              </li>
            ))}
          </ul>
        </section>
      </div>

      {/* Equipment Management Section - Bottom Middle */}
      <section className="crm-section crm-equipment crm-equipment-bottom">
        <h2>Other Equipment Management</h2>
        <div className="crm-form">
          <input type="text" placeholder="Equipment Name" value={newEquipment.name} onChange={(e) => handleInputChange(setNewEquipment, 'name', e.target.value)} />
          <input type="number" placeholder="Quantity" value={newEquipment.quantity} onChange={(e) => handleInputChange(setNewEquipment, 'quantity', e.target.value)} min="0"/>
          <textarea placeholder="Notes" value={newEquipment.notes} onChange={(e) => handleInputChange(setNewEquipment, 'notes', e.target.value)} />
          <button onClick={() => addItem('equipment')} disabled={loading}>Add Equipment</button>
        </div>
        <ul className="crm-list">
          {equipmentList.map((equip, index) => (
            <li key={equip.id || index}>
              {equip.name} (Qty: {equip.quantity})
              <button onClick={() => removeItem('equipment', index)} className="crm-remove-btn" disabled={loading}>Remove</button>
            </li>
          ))}
        </ul>
      </section>
      
      <div className="crm-actions">
        {saveMessage && <p className="crm-save-message">{saveMessage}</p>}
        <button onClick={handleSaveResources} className="crm-save-all-btn" disabled={loading}>
          {loading ? 'Saving...' : 'Save All Resources'}
        </button>
      </div>
    </div>
  );
};

export default CampResourceManagementPage;
