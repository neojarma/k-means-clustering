// auth.js
document.addEventListener("DOMContentLoaded", function () {
    // Check if the user is logged in
    const isLoggedIn = localStorage.getItem('isLoggedIn');

    // If not logged in, redirect to the login page
    if (!isLoggedIn) {
        window.location.href = '../pages/sign-in.html';
    }
});