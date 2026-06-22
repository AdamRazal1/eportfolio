/* ============================================================
   Adam Razali — Portfolio
   Theme toggle, mobile nav, scroll reveal, navbar shadow.
   ============================================================ */
(function () {
  "use strict";

  /* ---------- Theme (light/dark) with persistence ---------- */
  var root = document.documentElement;
  var toggle = document.getElementById("themeToggle");
  var saved = localStorage.getItem("theme");
  var prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  var initial = saved || (prefersDark ? "dark" : "light");
  root.setAttribute("data-theme", initial);

  if (toggle) {
    toggle.addEventListener("click", function () {
      var next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      localStorage.setItem("theme", next);
    });
  }

  /* ---------- Mobile nav ---------- */
  var burger = document.getElementById("navBurger");
  var links = document.getElementById("navLinks");
  function closeMenu() {
    if (links) links.classList.remove("is-open");
    if (burger) burger.classList.remove("is-open");
  }
  if (burger && links) {
    burger.addEventListener("click", function () {
      links.classList.toggle("is-open");
      burger.classList.toggle("is-open");
    });
    links.querySelectorAll("a").forEach(function (a) {
      a.addEventListener("click", closeMenu);
    });
  }

  /* ---------- Navbar border on scroll ---------- */
  var nav = document.getElementById("nav");
  function onScroll() {
    if (!nav) return;
    nav.classList.toggle("is-scrolled", window.scrollY > 8);
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  /* ---------- Scroll reveal ---------- */
  function reveal() {
    var hero = document.querySelectorAll(".fade-up");
    hero.forEach(function (el, i) {
      setTimeout(function () { el.classList.add("in"); }, 80 * i);
    });

    var targets = document.querySelectorAll(
      ".section__title, .section__lead, .about, .skill-group, .project, .term, .contact__card"
    );
    targets.forEach(function (el) { el.classList.add("reveal"); });

    if (!("IntersectionObserver" in window)) {
      targets.forEach(function (el) { el.classList.add("in"); });
      return;
    }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add("in");
          io.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12, rootMargin: "0px 0px -40px 0px" });
    targets.forEach(function (el) { io.observe(el); });
  }
  reveal();

  /* ---------- Footer year ---------- */
  var year = document.getElementById("year");
  if (year) year.textContent = new Date().getFullYear();
})();
