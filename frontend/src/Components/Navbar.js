import React from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';

function Navbar() {
  const navigate = useNavigate();
  const location = useLocation();
  const pathname = location.pathname;

  const loggedIn = Boolean(localStorage.getItem('token')); // Check if user is logged in

  // Handle logout and redirect to login
  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('email');
    navigate('/login'); // Redirect to login page
  };

  // Show Dashboard if logged in and current page is not Dashboard
  const showDashboardButton = loggedIn && pathname !== '/dashboard';

  return (
    <>
      <style>{`
        .navbar-dark .dropdown-menu {
          background: linear-gradient(135deg, #212529 0%, #343a40 100%);
          border: none;
          box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        }
        .navbar-dark .dropdown-item {
          color: #f8f9fa;
        }
        .navbar-dark .dropdown-item:hover,
        .navbar-dark .dropdown-item:focus {
          background-color: rgba(255,255,255,0.1);
        }
        .navbar .nav-item.dropdown:hover > .dropdown-menu {
          display: block;
        }
      `}</style>

      <nav className="navbar navbar-expand-lg navbar-dark bg-dark shadow-lg">
        <div className="container">
         
            <img src="/LogoLung.jpg" alt="logo" width="60px" style={{ borderRadius: '12px' }} />
         
          <button
            className="navbar-toggler"
            type="button"
            data-bs-toggle="collapse"
            data-bs-target="#mainNavbar"
            aria-controls="mainNavbar"
            aria-expanded="false"
            aria-label="Toggle navigation"
          >
            <span className="navbar-toggler-icon" />
          </button>

          <div className="collapse navbar-collapse" id="mainNavbar">
            <ul className="navbar-nav ms-auto align-items-center">
              {/* Home Button: Logs out the user */}
              <li className="nav-item mx-2">
                <Link className="nav-link" to="/" onClick={handleLogout}>
                  Home
                </Link>
              </li>

              {/* Show Dashboard if logged in and not on the Dashboard page */}
              {showDashboardButton && (
                <li className="nav-item mx-2">
                  <Link className="nav-link" to="/dashboard">Dashboard</Link>
                </li>
              )}

              {/* About links */}
              <li className="nav-item mx-2">
                <Link className="nav-link" to="/about-us">About Us</Link>
              </li>
              <li className="nav-item mx-2">
                <Link className="nav-link" to="/about-algorithm">About Algorithm</Link>
              </li>

              {/* Account menu for logged in users */}
              {loggedIn ? (
                <li className="nav-item dropdown mx-2">
                  <button
                    className="nav-link dropdown-toggle bg-transparent border-0"
                    id="accountDropdown"
                    data-bs-toggle="dropdown"
                    aria-expanded="false"
                  >
                    Account
                  </button>
                  <ul className="dropdown-menu dropdown-menu-end" aria-labelledby="accountDropdown">
                    <li>
                      <Link className="dropdown-item" to="/previous-actions">
                        Previous Actions
                      </Link>
                    </li>
                    <li><hr className="dropdown-divider" /></li>
                    <li>
                      <button className="dropdown-item text-danger" onClick={handleLogout}>
                        Logout
                      </button>
                    </li>
                  </ul>
                </li>
              ) : (
                <li className="nav-item mx-2">
                  {pathname === '/login' ? (
                    <Link className="btn btn-outline-light" to="/">Register</Link>
                  ) : (
                    <Link className="btn btn-outline-light" to="/login">Login</Link>
                  )}
                </li>
              )}
            </ul>
          </div>
        </div>
      </nav>
    </>
  );
}

export default Navbar;

