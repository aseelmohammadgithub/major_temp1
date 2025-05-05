// src/pages/PreviousActionsPage.js

import React from 'react';
import Navbar from '../Components/Navbar';
import PreviousActions from '../Components/PreviousActions';

function PreviousActionsPage() {
  return (
    <>
      <Navbar loggedIn={true} />
      <PreviousActions />
    </>
  );
}

export default PreviousActionsPage;
