--
-- PostgreSQL database dump
--

\restrict IQcvBZvGgddNbHWnHX0Ambs8G9dlaYCX09zrgKHgdEr25qYPx4rUUAPeelBF0ad

-- Dumped from database version 16.14 (Ubuntu 16.14-0ubuntu0.24.04.1)
-- Dumped by pg_dump version 16.14 (Ubuntu 16.14-0ubuntu0.24.04.1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

ALTER TABLE ONLY public.unfaelle DROP CONSTRAINT unfaelle_pkey;
ALTER TABLE public.unfaelle ALTER COLUMN id DROP DEFAULT;
DROP SEQUENCE public.unfaelle_id_seq;
DROP TABLE public.unfaelle_dataset;
DROP TABLE public.unfaelle;
SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: unfaelle; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.unfaelle (
    id integer NOT NULL,
    roh_beschreibung text NOT NULL,
    auto_marke character varying(50),
    auto_baujahr integer,
    beschaedigte_teile text,
    interner_schadensgrad character varying(20),
    vorhergesagte_reparaturkosten numeric(10,2),
    ist_totalschaden boolean,
    erstellt_am timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: unfaelle_dataset; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.unfaelle_dataset (
    auto_make_model text,
    auto_year bigint,
    incident_severity text,
    collision_type text,
    claim_amount_eur double precision,
    generated_description text
);


--
-- Name: unfaelle_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.unfaelle_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: unfaelle_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.unfaelle_id_seq OWNED BY public.unfaelle.id;


--
-- Name: unfaelle id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.unfaelle ALTER COLUMN id SET DEFAULT nextval('public.unfaelle_id_seq'::regclass);


--
-- Name: unfaelle unfaelle_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.unfaelle
    ADD CONSTRAINT unfaelle_pkey PRIMARY KEY (id);


--
-- Name: SCHEMA public; Type: ACL; Schema: -; Owner: -
--

GRANT ALL ON SCHEMA public TO projekt_user;


--
-- Name: TABLE unfaelle; Type: ACL; Schema: public; Owner: -
--

GRANT ALL ON TABLE public.unfaelle TO projekt_user;


--
-- Name: SEQUENCE unfaelle_id_seq; Type: ACL; Schema: public; Owner: -
--

GRANT ALL ON SEQUENCE public.unfaelle_id_seq TO projekt_user;


--
-- PostgreSQL database dump complete
--

\unrestrict IQcvBZvGgddNbHWnHX0Ambs8G9dlaYCX09zrgKHgdEr25qYPx4rUUAPeelBF0ad

